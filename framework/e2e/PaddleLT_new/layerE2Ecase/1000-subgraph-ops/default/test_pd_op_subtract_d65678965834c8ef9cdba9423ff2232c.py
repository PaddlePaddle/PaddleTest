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



class PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_721630adc70c95896527591cc9fdbb05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_721630adc70c95896527591cc9fdbb05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_721630adc70c95896527591cc9fdbb05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_721630adc70c95896527591cc9fdbb05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_721630adc70c95896527591cc9fdbb05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_721630adc70c95896527591cc9fdbb05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_721630adc70c95896527591cc9fdbb05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_721630adc70c95896527591cc9fdbb05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2530a4e4625af688f4e3de9927836b64(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6, 1, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ae5fb2d0ee9b7b818f7b4f0c5393024e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2530a4e4625af688f4e3de9927836b64
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.02815207839012146, 0.018858909606933594]], [[0.385414183139801, -0.18645644187927246]], [[0.3229190707206726, 0.46594446897506714]], [[0.3456408977508545, -0.31946736574172974]], [[-0.29189950227737427, 0.2838500738143921]], [[-0.24781763553619385, 0.47005563974380493]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[-0.416607141494751, 0.4868149161338806]], [[0.3664243817329407, 0.25259023904800415]], [[-0.2397032082080841, 0.40948301553726196]], [[-0.047472745180130005, -0.44494539499282837]], [[0.46502143144607544, -0.3843851387500763]], [[-0.3654167056083679, 0.47668784856796265]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_14c48b964af1cd0d085125ca2541a4e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2530a4e4625af688f4e3de9927836b64
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.20604151487350464, -0.20769765973091125]], [[0.0012865066528320312, -0.36628061532974243]], [[0.21106261014938354, -0.23206135630607605]], [[-0.30802029371261597, -0.03535628318786621]], [[0.14247292280197144, 0.1793147325515747]], [[0.3005167245864868, 0.21434420347213745]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[-0.416607141494751, 0.4868149161338806]], [[0.3664243817329407, 0.25259023904800415]], [[-0.2397032082080841, 0.40948301553726196]], [[-0.047472745180130005, -0.44494539499282837]], [[0.46502143144607544, -0.3843851387500763]], [[-0.3654167056083679, 0.47668784856796265]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class PrimitiveOp_38e60ad0c3265f689e713b6b70f444bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 21824, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6, 1, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c41ef5987568bcf406afc5eb3c672579(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38e60ad0c3265f689e713b6b70f444bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 21824, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[-0.0214555561542511, -0.2133394479751587]], [[-0.2730659246444702, -0.48764875531196594]], [[-0.4760924279689789, -0.3975455164909363]], [[0.2721167802810669, -0.14962509274482727]], [[0.2108592987060547, -0.4039212465286255]], [[0.4944922924041748, -0.19539695978164673]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class PrimitiveOp_bfa3b4dcb0b3a0e55b5821ad7c27cbbb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 100, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 100, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8899f76134e74e4bc40d1814f72eedaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfa3b4dcb0b3a0e55b5821ad7c27cbbb
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3a37b23951ff5101fe91834373112dd7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 100, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 4, 100, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d3448cde4f62d9499e9a52e8243e86d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a37b23951ff5101fe91834373112dd7
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 100, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6131f1e087763b89c98fd9eaba27d305(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[100, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_26bf2a966564312b1fb268edd62efe45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6131f1e087763b89c98fd9eaba27d305
    def get_inputs(self):
        return [
            paddle.uniform([100, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([100, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1546d5060eb5e7f91c606a3154f7d7bd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 1, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7bf7a700814936e089f43257ecf1542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1546d5060eb5e7f91c606a3154f7d7bd
    def get_inputs(self):
        return [
            paddle.uniform([100, 1, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[1.3296499252319336, 10.861931800842285, -0.5463329553604126, -0.8369226455688477], [-1.479607105255127, -0.6961761116981506, 0.7482630014419556, -18.51576805114746]], dtype='float32').reshape([2, 4]),
        ]


class PrimitiveOp_f31d773a88065c75d61f82477b779d63(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1025, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f8ec5f086f5247de37a827f2722ccd4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f31d773a88065c75d61f82477b779d63
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1025, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1c218da9b45c54722e8f2702a335865a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1025, 1025], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 12, 1025, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8fb8cd60ded0544b92562aba50a629e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c218da9b45c54722e8f2702a335865a
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 1025, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_75073f8acf4a44b30d50e7a9b6b01e5a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15200, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[15200, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_099885e5f2bf39f74d03621c5c71a1a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75073f8acf4a44b30d50e7a9b6b01e5a
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8b88a0911757050cc1660428654c44c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9efb099ba0da58e3b1d7c20913c09249(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1508, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_956868f03a66db2ba40241b5cc43e189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_956868f03a66db2ba40241b5cc43e189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_956868f03a66db2ba40241b5cc43e189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_956868f03a66db2ba40241b5cc43e189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_956868f03a66db2ba40241b5cc43e189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_956868f03a66db2ba40241b5cc43e189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_956868f03a66db2ba40241b5cc43e189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_956868f03a66db2ba40241b5cc43e189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_956868f03a66db2ba40241b5cc43e189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_956868f03a66db2ba40241b5cc43e189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_956868f03a66db2ba40241b5cc43e189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_28893411881ab0c201d00b2effafe770(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3024, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3024, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_42b43b179161b494aeac4856bdd94a4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28893411881ab0c201d00b2effafe770
    def get_inputs(self):
        return [
            paddle.uniform([3024, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3024, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ccd1f9a8924543a5f9c8c87c15c9e66e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[3024, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fec0195a2aff35fe3d803815239871a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccd1f9a8924543a5f9c8c87c15c9e66e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3024, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9efb099ba0da58e3b1d7c20913c09249(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1508, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03efbf583dbd12f0506b8fe4a09a6efc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2377, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1690ee0e66127e2ad2eaf5b4c0a2f409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1690ee0e66127e2ad2eaf5b4c0a2f409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1690ee0e66127e2ad2eaf5b4c0a2f409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1690ee0e66127e2ad2eaf5b4c0a2f409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1690ee0e66127e2ad2eaf5b4c0a2f409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1690ee0e66127e2ad2eaf5b4c0a2f409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1690ee0e66127e2ad2eaf5b4c0a2f409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1690ee0e66127e2ad2eaf5b4c0a2f409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1690ee0e66127e2ad2eaf5b4c0a2f409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1690ee0e66127e2ad2eaf5b4c0a2f409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1690ee0e66127e2ad2eaf5b4c0a2f409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d0bccc3bedd14bdf0e2dad1e1f7dbee5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4725, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4725, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c5cb52b14fb741d676eb1ed93f6a6dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0bccc3bedd14bdf0e2dad1e1f7dbee5
    def get_inputs(self):
        return [
            paddle.uniform([4725, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4725, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bdbccc0bd9d4fa6fab63f7856283dda3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[4725, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a8b1742181645a365afdd2d3b9b5e02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdbccc0bd9d4fa6fab63f7856283dda3
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4725, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03efbf583dbd12f0506b8fe4a09a6efc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2377, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_85f8d145f57480c28b005bd43f1f1a7a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1024, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_91dca2038b8ef8d22e73b7f6baac75ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85f8d145f57480c28b005bd43f1f1a7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1024, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ab7278de0bfd314eb4d8dfd8fc87222c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1024, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8, 1024, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7749b6386f3d44d6dbf21fc5378de926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab7278de0bfd314eb4d8dfd8fc87222c
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 1024, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_66653002574c8cbef701d1e51b77e0e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 2304, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 2304, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca54a4e4f13dc88d88ee75c0b75fe461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66653002574c8cbef701d1e51b77e0e2
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 2304, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0cb5bc7575b10cab7a5681e75c2ab059(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 3136, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 3136, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a145880bbd22aff10323e75e568e226e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cb5bc7575b10cab7a5681e75c2ab059
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e8e67cda4c004970d7e5da1ba47481fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 3136, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 3136, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e94d8f4bbad8687b70f1f5eb97ff700c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e67cda4c004970d7e5da1ba47481fb
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_733c155888a4588c13781126b76103a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 196, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b2504e6c19ad4850aa8afad3437f488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_733c155888a4588c13781126b76103a8
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_674c962dcbbeff6af94ff9ebb91ba7aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_51ef665bddcca8edc0191885a77599df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_674c962dcbbeff6af94ff9ebb91ba7aa
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5562693c80f4b6c017b21160400867ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3bd71b1091a034820703724d3bddf585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3bd71b1091a034820703724d3bddf585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3bd71b1091a034820703724d3bddf585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3bd71b1091a034820703724d3bddf585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3bd71b1091a034820703724d3bddf585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3bd71b1091a034820703724d3bddf585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3bd71b1091a034820703724d3bddf585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0615e0f648ce468e294d7ab966bb6dd5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 197, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[54, 197, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d0682d6bba241945062b0d8f07380b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0615e0f648ce468e294d7ab966bb6dd5
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 197, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_87554ef2fdba0f34fc327b753616e4b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 197, 197], dtype='float32'),
            paddle.static.InputSpec(shape=[54, 3, 197, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d6703299a0998b994c31e99c751f9fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87554ef2fdba0f34fc327b753616e4b7
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 197], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 3, 197, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cea6fb04c6ab3a7c9d9cf93206b2bb68(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 784, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 784, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_809fdcfe05bb9dbf3dd2db42991ace51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cea6fb04c6ab3a7c9d9cf93206b2bb68
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1318caf4323f0ef693ca24eed5e1198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2825867533683777], [-0.4766601324081421], [-0.12349918484687805], [-0.017541974782943726]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.029543638229370117], [0.02459317445755005], [-0.020875394344329834], [-0.3715863823890686]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_ffa7b2ca4776d4fe8dc31443af0f7b13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1602063775062561], [-0.14480668306350708], [0.34999436140060425], [-0.4619288444519043]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.016034811735153198], [-0.04461756348609924], [-0.13033828139305115], [0.46304792165756226]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_b8c0cd781ec6d7b72fa25932352195d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4195559024810791], [0.258736252784729], [0.4700632691383362], [-0.007235139608383179]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.42702656984329224], [-0.03797486424446106], [-0.32467591762542725], [-0.3715863823890686]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_2233f9a9674f98a3543cec95046744b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1602063775062561], [-0.14480668306350708], [0.40289902687072754], [-0.3244452476501465]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.016034811735153198], [-0.04461756348609924], [-0.13033828139305115], [0.1816677451133728]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_9f10dc9eb0a12b8ccb502758facb1a6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2825867533683777], [-0.4766601324081421], [-0.12349918484687805], [-0.017541974782943726]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.029543638229370117], [0.02459317445755005], [-0.020875394344329834], [-0.43152952194213867]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_09f07a47ff3849193cc36c8661d130e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.38134509325027466], [0.24650567770004272], [0.34999436140060425], [-0.4619288444519043]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.29278564453125], [-0.1465194821357727], [-0.4884662628173828], [0.46304792165756226]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_422f76094b3f68262e0d0afeada601b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3596193790435791], [-0.22673238813877106], [0.33773860335350037], [-0.5673317313194275]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.05501023307442665], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_36891a227a16ccaf5babf40479f456f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4195559024810791], [0.258736252784729], [0.4700632691383362], [-0.007235139608383179]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.42702656984329224], [-0.03797486424446106], [-0.32467591762542725], [-0.43152952194213867]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_54ea410e86cfbcc66064cd0e38d51fb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.38134509325027466], [0.24650567770004272], [0.40289902687072754], [-0.3244452476501465]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.29278564453125], [-0.1465194821357727], [-0.4884662628173828], [0.1816677451133728]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_da4efb66a0091c9b7f7b5630db86ae95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.5707072615623474], [0.1166149377822876], [0.7084029316902161], [-0.21474090218544006]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.30460914969444275], [-0.22673238813877106], [0.33773860335350037], [-0.5673317313194275]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_9489b68a8531f2d400d35463bd4feeaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1805928498506546], [-0.0], [0.0], [-0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.46626025438308716], [2.9442825317382812], [0.5232394337654114], [-1.6419360637664795]], dtype='float32').reshape([4, 1]),
        ]


class PrimitiveOp_5dfcedf316eacbbadf7dd6cacaded896(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ef18bf8c5fc8b1787c744cf570facdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dfcedf316eacbbadf7dd6cacaded896
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.26601147651672363]], [[0.4025825262069702]], [[0.14946210384368896]], [[-0.1720055341720581]], [[-0.17647510766983032]], [[0.38267022371292114]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.36880454421043396]], [[0.5379533171653748]], [[0.7502270340919495]], [[0.36781883239746094]], [[0.4300179183483124]], [[0.6875841021537781]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_7c0a735520913fd48b7e1ca1127708c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dfcedf316eacbbadf7dd6cacaded896
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.38691067695617676]], [[-0.299028217792511]], [[0.1920897364616394]], [[-0.36260396242141724]], [[0.08917808532714844]], [[0.25575244426727295]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.4420416057109833]], [[0.7622336149215698]], [[0.7119382619857788]], [[0.5102642178535461]], [[0.4841826260089874]], [[0.3682727515697479]]], dtype='float32').reshape([6, 1, 1]),
        ]


class PrimitiveOp_b56e173860ecca42b7de9bf0c299934f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 7581, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_827528a2146e9f9c7fc097260c4a6972(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b56e173860ecca42b7de9bf0c299934f
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 7581, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            paddle.static.InputSpec(shape=[6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_06449a2d95f550aba04316959aa93e1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.13945317268371582, 0.2671630382537842, -0.120673269033432, -0.19917404651641846, 0.26317787170410156, 0.17146217823028564], dtype='float32').reshape([6]),
            paddle.to_tensor([0.32623517513275146, -0.24192336201667786, 0.27793484926223755, 0.19273710250854492, 0.4619840383529663, 0.057091474533081055], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_67f155a0f200a661d74d248a11890b96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.21393254399299622, 0.3412080407142639, 0.2899574637413025, -0.25859230756759644, -0.26445272564888, -0.4373926520347595], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.48779433965682983, 0.34922271966934204, -0.4023532271385193, 0.3019695281982422, 0.06090348958969116, 0.11269927024841309], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_e2654555e0394e8a1e1289373fbb0fcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.270524263381958, -0.20599150657653809, -0.06320270895957947, -0.2330784797668457, -0.28073909878730774, 0.2532276511192322], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.03316903114318848, 0.37095922231674194, 0.48807185888290405, 0.02101588249206543, -0.059602320194244385, 0.2165842056274414], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_a8bce08b2719b98bff8717c755825fc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.19772139191627502, 0.4174601435661316, 0.029941082000732422, -0.15820688009262085, -0.4392582178115845, -0.4687579572200775], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.03374040126800537, -0.1420654058456421, -0.10500526428222656, -0.35081708431243896, -0.40645480155944824, -0.12897560000419617], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_de47b26fb1a997691b6201fb178ffe42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.270524263381958, -0.20599150657653809, -0.06320270895957947, -0.2330784797668457, -0.28073909878730774, 0.17146217823028564], dtype='float32').reshape([6]),
            paddle.to_tensor([0.32623517513275146, 0.37095922231674194, 0.48807185888290405, 0.19273710250854492, 0.4619840383529663, 0.2165842056274414], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_b27fcc2ece8c6a63d032b6f29b04a0dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.21393254399299622, 0.34922271966934204, 0.029941082000732422, -0.15820688009262085, -0.4392582178115845, -0.4687579572200775], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.03374040126800537, 0.34922271966934204, -0.10500526428222656, 0.3019695281982422, 0.06090348958969116, 0.11269927024841309], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_2c2d3b1cb217f39611e7e879b922f192(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.32623517513275146, 0.2671630382537842, 0.27793484926223755, 0.19273710250854492, 0.4619840383529663, 0.17146217823028564], dtype='float32').reshape([6]),
            paddle.to_tensor([0.32623517513275146, -0.24192336201667786, 0.27793484926223755, 0.19273710250854492, 0.4619840383529663, 0.057091474533081055], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_45ae80e029e8e2ea857ba909a7d87e50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.21393254399299622, 0.34922271966934204, 0.2899574637413025, 0.3019695281982422, 0.06090348958969116, 0.11269927024841309], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.48779433965682983, 0.34922271966934204, -0.4023532271385193, 0.3019695281982422, 0.06090348958969116, 0.11269927024841309], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_f64ca6a5676ebb1937a07dbb594ea99e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.038921747356653214, -0.32281866669654846, -0.07439248263835907, -0.04894116520881653, 0.0072540417313575745, -0.01245079655200243], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, -0.0, -0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_9a00aa36946d4a67af9aa5a49ebc21c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23284417390823364, 0.012619838118553162, 0.07863079011440277, -0.0032184720039367676, 0.36258095502853394, 0.11427682638168335], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.15184664726257324, 0.08248385787010193, 0.2124345749616623, -0.10603129863739014, -0.17017070949077606, 0.2349059283733368], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_73f5bc16a52bd7edfb6ffab8e15886b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3508634567260742, 0.345215380191803, -0.0561978816986084, 0.021688610315322876, -0.10177461802959442, -0.16234669089317322], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.1157308965921402, 0.13769736886024475, -0.03753209114074707, -0.2545119822025299, -0.42285650968551636, -0.29886677861213684], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ab0985b8121015de7cb4fa3f44e8fce5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.32623517513275146, 0.2671630382537842, 0.27793484926223755, 0.19273710250854492, 0.4619840383529663, 0.2532276511192322], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.03316903114318848, -0.24192336201667786, 0.27793484926223755, 0.02101588249206543, -0.059602320194244385, 0.057091474533081055], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_941787058a4e5400296c87c967506ea2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.19772139191627502, 0.4174601435661316, 0.2899574637413025, 0.3019695281982422, 0.06090348958969116, 0.11269927024841309], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.48779433965682983, -0.1420654058456421, -0.4023532271385193, -0.35081708431243896, -0.40645480155944824, -0.12897560000419617], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_df6081817f95ff94a7bcab44c0c8f20a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.9662259817123413, -0.8007296919822693, -1.330727458000183, -0.9221782088279724, 1.4235303401947021, -0.10742868483066559], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.598563551902771, -1.5550544261932373, -0.5224088430404663, 0.6101484894752502, 0.5484987497329712, -0.20499153435230255], dtype='float32').reshape([6]),
        ]


class PrimitiveOp_ddcc881c03fd6c994972310f815a76b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[47, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[47, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_81aecda33808ed2f91bc4dd7f25d32b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddcc881c03fd6c994972310f815a76b3
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([47, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_20b61133f307c883835c65242e70f6a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2100, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66355d29b0c68d715a2b32de7e3b00c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20b61133f307c883835c65242e70f6a8
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2100, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cac5b103f5b30e2b736b268f78421668(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 49, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20bd16966e6ab9b1495762af0151051b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cac5b103f5b30e2b736b268f78421668
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3ee85a9a59cd15f4a297868d77aa7453(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 49, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b2c1714ff69c46636eb8bd6c69c7735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ee85a9a59cd15f4a297868d77aa7453
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_63c8dbd78d2b4b958084abc34dd9c748(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4, 1, 50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c0880f3b213aade90e50c249f9e2e4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63c8dbd78d2b4b958084abc34dd9c748
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4, 1, 50, 76], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b9a64d4003842f319715ec8c702cfde5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 640, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 640, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_36b6e5352e5a01e026d66e2ca05ce451(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9a64d4003842f319715ec8c702cfde5
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 640, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fd57222527203885a891498d7c9b1024(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 640, 640], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 2, 640, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6004a8e96c13ed1bca3ee50a462d05b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd57222527203885a891498d7c9b1024
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 640], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 2, 640, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_20bd16966e6ab9b1495762af0151051b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cac5b103f5b30e2b736b268f78421668
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3929f14f147410bb4e712b4b16c1521a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3929f14f147410bb4e712b4b16c1521a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3929f14f147410bb4e712b4b16c1521a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3929f14f147410bb4e712b4b16c1521a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3929f14f147410bb4e712b4b16c1521a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3929f14f147410bb4e712b4b16c1521a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3929f14f147410bb4e712b4b16c1521a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4b7bf90f773aa50024593645a0952115(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a0bf4846098a19b4a316fe88f8a4017c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0bf4846098a19b4a316fe88f8a4017c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0bf4846098a19b4a316fe88f8a4017c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0bf4846098a19b4a316fe88f8a4017c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0bf4846098a19b4a316fe88f8a4017c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0bf4846098a19b4a316fe88f8a4017c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0bf4846098a19b4a316fe88f8a4017c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0bf4846098a19b4a316fe88f8a4017c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ef80528f0a858e03e82f793ae6527930(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[300, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_95088e053d10c565104638b42fac788c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef80528f0a858e03e82f793ae6527930
    def get_inputs(self):
        return [
            paddle.uniform([300, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([300, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_29eb514882c92ab3a86517239fc1edf9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 1, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6cefd463d7108518780b03092f302651(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29eb514882c92ab3a86517239fc1edf9
    def get_inputs(self):
        return [
            paddle.uniform([300, 1, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[-55.30966567993164, 0.19226625561714172, 0.11763492226600647, -1.465273141860962], [3.514228105545044, 4.117565155029297, -0.5635074973106384, -0.36819878220558167]], dtype='float32').reshape([2, 4]),
        ]


class PrimitiveOp_f789fea4ace583837c4ce54ad5c69b47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4, 1, 100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7cf778babd5f887ffdd2981c06fafbda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f789fea4ace583837c4ce54ad5c69b47
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4, 1, 100, 152], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4603fe0ce7ef18f56a1834167dd2c8c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21, 16384], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 21, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b0830132ccaf3702a90a76fa58abe0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4603fe0ce7ef18f56a1834167dd2c8c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 16384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[0.49999362230300903], [0.49997013807296753], [0.49998801946640015], [0.49990522861480713], [0.4998725652694702], [0.4999591112136841], [0.49994707107543945], [0.4999229907989502], [0.4999859929084778], [0.4999842047691345], [0.49988502264022827], [0.49995744228363037], [0.4999678134918213], [0.4999958276748657], [0.49987900257110596], [0.49991631507873535], [0.49997377395629883], [0.4998912215232849], [0.49995070695877075], [0.4999926686286926], [0.4999288320541382]]], dtype='float32').reshape([1, 21, 1]),
        ]


class PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9544a1dc96ec4c05e13a86968e8c439d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9544a1dc96ec4c05e13a86968e8c439d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9544a1dc96ec4c05e13a86968e8c439d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9544a1dc96ec4c05e13a86968e8c439d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9544a1dc96ec4c05e13a86968e8c439d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9544a1dc96ec4c05e13a86968e8c439d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9544a1dc96ec4c05e13a86968e8c439d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a145880bbd22aff10323e75e568e226e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cb5bc7575b10cab7a5681e75c2ab059
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4dd958c456eed18a80e7a0ddf3ff1f9e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 200, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 200, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e0bb79729fce3647e0ae6e600b14559c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dd958c456eed18a80e7a0ddf3ff1f9e
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 200, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fcfd71b006311c96bc64890082e78e0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 200, 200], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 2, 200, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_494f00c6c0e13f64a3b6d1a183a14e73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fcfd71b006311c96bc64890082e78e0b
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 2, 200, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d6eff661b5176970610103901c79f9b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 49, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_44d53373fadf4732b0f93853f02c9d5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6eff661b5176970610103901c79f9b1
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70aba1b143bef2ac7d30ded668c49441(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10710150003433228], [0.3015877604484558], [-0.2885453701019287], [-0.31574276089668274], [-0.39915066957473755], [-0.2773142158985138]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.35990118980407715], [0.4532351493835449], [0.3253195881843567], [0.23642903566360474], [0.22819304466247559], [0.10179847478866577]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_9141e860d695a40a2fdac86828ab1aa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.4103016257286072], [-0.3500272035598755], [-0.0469474196434021], [-0.09059321880340576], [-0.05198678374290466], [-0.2040221095085144]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.36881452798843384], [0.446461021900177], [0.25204068422317505], [0.18123924732208252], [-0.04423898458480835], [0.028705358505249023]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_a51529103506ea60920c9916b56ca295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2814381718635559], [0.3015877604484558], [-0.2885453701019287], [-0.31574276089668274], [0.2026931643486023], [-0.0327763557434082]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.35990118980407715], [0.4532351493835449], [0.3253195881843567], [-0.18913787603378296], [0.22819304466247559], [0.10179847478866577]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_55cb88d5db2887f59be8c37136fa46d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.45480841398239136], [-0.3500272035598755], [-0.0469474196434021], [0.07794475555419922], [0.19924408197402954], [-0.2040221095085144]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.36881452798843384], [0.446461021900177], [0.25204068422317505], [0.18123924732208252], [-0.04423898458480835], [0.028705358505249023]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_7c03ca81ce3759488731d164723e4de5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10710150003433228], [0.4202950596809387], [0.20075774192810059], [-0.24555674195289612], [-0.39915066957473755], [-0.2773142158985138]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.34053710103034973], [0.2731626033782959], [-0.39370104670524597], [0.23642903566360474], [-0.17116469144821167], [-0.068158358335495]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_40e0956390ce384fd6b569b7bad61a25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.4103016257286072], [-0.2566860318183899], [0.030750691890716553], [-0.09059321880340576], [-0.05198678374290466], [0.11784225702285767]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.4686683714389801], [0.3912338614463806], [-0.45743855834007263], [-0.008816301822662354], [-0.15691471099853516], [-0.17567184567451477]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_d70c8c484dda8cc7218c7fb52a5c2893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.03849673643708229], [0.02545531839132309], [0.47374671697616577], [0.0524928979575634], [-0.030130885541439056], [-0.03007093444466591]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_b7d3dc5f664b9633051227b09aacc6b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2814381718635559], [0.4202950596809387], [0.20075774192810059], [-0.24555674195289612], [0.2026931643486023], [-0.0327763557434082]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.34053710103034973], [0.2731626033782959], [-0.39370104670524597], [-0.18913787603378296], [-0.17116469144821167], [-0.068158358335495]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_ed42b94f923390e1a20f058babc4321f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.45480841398239136], [-0.2566860318183899], [0.030750691890716553], [0.07794475555419922], [0.19924408197402954], [0.11784225702285767]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.4686683714389801], [0.3912338614463806], [-0.45743855834007263], [-0.008816301822662354], [-0.15691471099853516], [-0.17567184567451477]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_ffefbfcab00ec23a866e84ba28160863(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.5743797421455383], [-0.09533004462718964], [0.2902083992958069], [-0.004894960671663284], [0.1331527680158615], [0.010385116562247276]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.03849673643708229], [0.02545531839132309], [0.47374671697616577], [0.0524928979575634], [-0.030130885541439056], [-0.03007093444466591]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_d16c5850d688752fb4b184349a6e8057(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [0.0], [0.0], [0.0], [-0.0], [-0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[1.0670231580734253], [1.2670230865478516], [-0.6324362754821777], [11.723865509033203], [1.226288080215454], [3.8955795764923096]], dtype='float32').reshape([6, 1]),
        ]


class PrimitiveOp_927a71b764361148b0a921336bec5333(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9261, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d319a22a421002ba97586f1462a49b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_927a71b764361148b0a921336bec5333
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9261, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9927cc6f9e052f94df4befead71a14f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 196, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c48802e6fe1109a80857a5f232e4acef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9927cc6f9e052f94df4befead71a14f9
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_77d5e089d8d1d89c52d957071f873a7e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4116, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c591d64fb8d375f9eceb1a706f1c98df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77d5e089d8d1d89c52d957071f873a7e
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4116, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_acf923d735a2711a1578a2716327c014(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3800, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[3800, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e6ca20ff1841fe2b4fd8d4ba0d74a3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acf923d735a2711a1578a2716327c014
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_473768ad5406d82abfbdd94c8502a0d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            paddle.static.InputSpec(shape=[4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f847dd64ddfd710720b559ad221ccbad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_473768ad5406d82abfbdd94c8502a0d7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
            paddle.to_tensor([0.20226573944091797, -0.19099652767181396, 0.3861161470413208, -0.4411441385746002], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_1a514b87b42937fce9e6ccce60f8e0ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_473768ad5406d82abfbdd94c8502a0d7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20226573944091797, -0.19099652767181396, 0.3861161470413208, -0.4411441385746002], dtype='float32').reshape([4]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([4]),
        ]


class PrimitiveOp_294b7063974dd0a73f2a3f686fb4d33d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21824, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 21824, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_164cfee86c8fdaffc7823fcf506402bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_294b7063974dd0a73f2a3f686fb4d33d
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6a40258a4deb7db4cf07129cd4d2b06b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 21824, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_381fc06df20f6bfacf90aaaf8e55082a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a40258a4deb7db4cf07129cd4d2b06b
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_381fc06df20f6bfacf90aaaf8e55082a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a40258a4deb7db4cf07129cd4d2b06b
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_488780642555bbf420383e7fb6ba9d19(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 49, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_879d8dec58de1d2fbee7c93f856dbae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488780642555bbf420383e7fb6ba9d19
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2da593dfe615c4063698f57af324953a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4, 1, 7, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_21a23c61f83c41184756ef8b702e0e50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2da593dfe615c4063698f57af324953a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4, 1, 7, 10], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8876d81ab777fa0cf2d24866b85ea4be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 60800, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 60800, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7689fd4d7e9412ef0875e5cd43d18634(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8876d81ab777fa0cf2d24866b85ea4be
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 60800, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_52d5008448878c67a6c2e468a48568d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 49, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_56d6f7b95c20539f60e6e01de098bffc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52d5008448878c67a6c2e468a48568d3
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e94d8f4bbad8687b70f1f5eb97ff700c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e67cda4c004970d7e5da1ba47481fb
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_30894a70778990ba804af814e417d17d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_35c5853473822729f000e399843e2039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30894a70778990ba804af814e417d17d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b38cea3f6f5c141b129a6d4ec2046a84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1a87515d713804f0946dc2bc4c24a38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b38cea3f6f5c141b129a6d4ec2046a84
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5c9935ad00e463f22cce5a3d36607f6b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 784, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 784, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f894fb8191dda138ef80fe732fac8c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9935ad00e463f22cce5a3d36607f6b
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1f492a007da4273c674695f4e88c6cbf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21504, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 21504, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df3b44aa0f46b92783f1b23075f36fb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f492a007da4273c674695f4e88c6cbf
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 21504, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e76424d8e783d99215bce247e73e5bfc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21504, 1, 91], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 21504, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d0bcf922431c20c80f7b297e383fe4f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e76424d8e783d99215bce247e73e5bfc
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 1, 91], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 21504, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b4367c13caf8c619ef448818b71b78e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 49, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_113f291148f8bc32b7236b768ae44a90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4367c13caf8c619ef448818b71b78e9
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_080a81e4a7b89445069bd25d1db2cd26(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 9216, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 9216, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_95d47ed588bcea580d7add676d09ba08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_080a81e4a7b89445069bd25d1db2cd26
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 9216, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e6ca20ff1841fe2b4fd8d4ba0d74a3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acf923d735a2711a1578a2716327c014
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c48802e6fe1109a80857a5f232e4acef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9927cc6f9e052f94df4befead71a14f9
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8b2c1714ff69c46636eb8bd6c69c7735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ee85a9a59cd15f4a297868d77aa7453
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e94d8f4bbad8687b70f1f5eb97ff700c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e67cda4c004970d7e5da1ba47481fb
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_49856d059716d3df67c5f17f94ba8e37(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 128, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d3d6047b2cf427ac8d3a0f54e10fba45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49856d059716d3df67c5f17f94ba8e37
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bed1e775ba8f7d7f86dd15571017dbbd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 128, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0082599b93cc425b92f0aef28cb07861(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed1e775ba8f7d7f86dd15571017dbbd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f68b329856f69b9bbf30c42a2fd0a85b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2204, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[2204, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b217e2544b0073962c5017695d63089a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f68b329856f69b9bbf30c42a2fd0a85b
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1b820b60b96edeb6b2c4753edd23b40b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            paddle.static.InputSpec(shape=[300], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5779acb2fa28261092846ecf7fc073e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b820b60b96edeb6b2c4753edd23b40b
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([300], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5779acb2fa28261092846ecf7fc073e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b820b60b96edeb6b2c4753edd23b40b
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([300], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_992c1c7449619f449fe668a2146b44c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aabb571e33c4a62dda3aa6ae2e3da4ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aabb571e33c4a62dda3aa6ae2e3da4ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aabb571e33c4a62dda3aa6ae2e3da4ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aabb571e33c4a62dda3aa6ae2e3da4ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aabb571e33c4a62dda3aa6ae2e3da4ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aabb571e33c4a62dda3aa6ae2e3da4ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aabb571e33c4a62dda3aa6ae2e3da4ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8b2c1714ff69c46636eb8bd6c69c7735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ee85a9a59cd15f4a297868d77aa7453
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e03f9187b8b7dac91f740b6d9ed74562(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 160, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 160, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af9b2c5ceda54843c3209c8ac38248ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e03f9187b8b7dac91f740b6d9ed74562
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 160, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f823b12447505d70b6bb8b883aa3b43b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 160, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 8, 160, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f3b2075f6a25fe8d858831d17840131(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f823b12447505d70b6bb8b883aa3b43b
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 8, 160, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e9a1123d79106ceb97abd981d8a72507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9a1123d79106ceb97abd981d8a72507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9a1123d79106ceb97abd981d8a72507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9a1123d79106ceb97abd981d8a72507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9a1123d79106ceb97abd981d8a72507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9a1123d79106ceb97abd981d8a72507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9a1123d79106ceb97abd981d8a72507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d051c7f30df6895d4d0b9ce8a122516f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 169, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 169, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_78aa199536e9e3ae06bcab1f57c2adb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d051c7f30df6895d4d0b9ce8a122516f
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 169, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_78aa199536e9e3ae06bcab1f57c2adb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d051c7f30df6895d4d0b9ce8a122516f
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 169, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b6c350fb07b219ab0b1860bf3b85187a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2015, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faf22d324e4d9b1c8c7ccba004c45f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faf22d324e4d9b1c8c7ccba004c45f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faf22d324e4d9b1c8c7ccba004c45f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faf22d324e4d9b1c8c7ccba004c45f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faf22d324e4d9b1c8c7ccba004c45f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faf22d324e4d9b1c8c7ccba004c45f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faf22d324e4d9b1c8c7ccba004c45f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faf22d324e4d9b1c8c7ccba004c45f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faf22d324e4d9b1c8c7ccba004c45f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faf22d324e4d9b1c8c7ccba004c45f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_faf22d324e4d9b1c8c7ccba004c45f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_50398eaf8f3d9b9c285466c003f7bb78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4116, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4116, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_99d2d5c0234b7757356679e379bcde16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50398eaf8f3d9b9c285466c003f7bb78
    def get_inputs(self):
        return [
            paddle.uniform([4116, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4116, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0808c9fcbc960a398d51addf53e19cab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[4116, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b00b3320963b03a2a16c4a8be000d2f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0808c9fcbc960a398d51addf53e19cab
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4116, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b6c350fb07b219ab0b1860bf3b85187a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2015, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_202983865b0e3a39199ea44ed1b4584f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9921b595bb1852afb4a7eb1e293d3f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9921b595bb1852afb4a7eb1e293d3f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9921b595bb1852afb4a7eb1e293d3f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9921b595bb1852afb4a7eb1e293d3f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9921b595bb1852afb4a7eb1e293d3f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9921b595bb1852afb4a7eb1e293d3f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9921b595bb1852afb4a7eb1e293d3f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9921b595bb1852afb4a7eb1e293d3f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7d642141b66a724f45dae974c1fd5dce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_126c3c3d01850f8f067cc2c80e4f96d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d642141b66a724f45dae974c1fd5dce
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3daa8676592f86ab4a3245e8d8e439c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[103, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[103, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5337235fc4c05e8c7465505fd4b3292d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3daa8676592f86ab4a3245e8d8e439c6
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([103, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2b7ef653997bed0eabe5f43e857422af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32768, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 32768, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_999745b86aee815cbf44a8f4ee44ae93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b7ef653997bed0eabe5f43e857422af
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 32768, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5a009f3831aa9518882502ffdcff086b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 320, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 320, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1cfd53e7ab5b4539c7c7dfcdc252c702(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a009f3831aa9518882502ffdcff086b
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1cdf7e55c209f160747e8650460062a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 320, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 4, 320, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59b9440736e839d50791cbaf62b0eef4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1cdf7e55c209f160747e8650460062a3
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ec31835e8609894bd0ca79587cdfc721(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_539c973cda41b214f69f861db982f127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_539c973cda41b214f69f861db982f127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_539c973cda41b214f69f861db982f127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_539c973cda41b214f69f861db982f127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_539c973cda41b214f69f861db982f127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_539c973cda41b214f69f861db982f127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_539c973cda41b214f69f861db982f127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_44b14ad5f465179c0238e126c886c102(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[950, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[950, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d689dbef7d7f50625a6ae6bc08ecd7dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44b14ad5f465179c0238e126c886c102
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1bfb56cd5191670ae5593396df788d44(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b29dd4d2c6675fcb5db56aa09ae22d36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b29dd4d2c6675fcb5db56aa09ae22d36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b29dd4d2c6675fcb5db56aa09ae22d36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b29dd4d2c6675fcb5db56aa09ae22d36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b29dd4d2c6675fcb5db56aa09ae22d36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b29dd4d2c6675fcb5db56aa09ae22d36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b29dd4d2c6675fcb5db56aa09ae22d36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b29dd4d2c6675fcb5db56aa09ae22d36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_38aa68f578e2965a292222c0287d8dc6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 1, 960, 960], dtype='float32'),
            paddle.static.InputSpec(shape=[2, 1, 960, 960], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c58e93619b79dc98793cf8aedba30154(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38aa68f578e2965a292222c0287d8dc6
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a430d95d326d4c5d5829f45991802fc7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8400, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a50552f7f8f1cea1c5bf32bf6daa6efc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a430d95d326d4c5d5829f45991802fc7
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8400, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_042c31aed200f7f6f007e22c52c71138(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            paddle.static.InputSpec(shape=[20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ac1a1c0dcae1d1d88914cd11d893e86f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_042c31aed200f7f6f007e22c52c71138
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
            paddle.to_tensor([0.05685281753540039, -0.3167967200279236, 0.05458754301071167, -0.48955684900283813, -0.04916509985923767, 0.3594948649406433, 0.14687931537628174, -0.3808799386024475, 0.4789818525314331, -0.4319356679916382, 0.42505037784576416, -0.13018929958343506, -0.27236491441726685, 0.2715114951133728, 0.26292920112609863, -0.1263890266418457, -0.37431198358535767, -0.2648187279701233, 0.22566699981689453, 0.06889081001281738], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_c6c827110e49ce5e9bfd19d4b46d06db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_042c31aed200f7f6f007e22c52c71138
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05685281753540039, -0.3167967200279236, 0.05458754301071167, -0.48955684900283813, -0.04916509985923767, 0.3594948649406433, 0.14687931537628174, -0.3808799386024475, 0.4789818525314331, -0.4319356679916382, 0.42505037784576416, -0.13018929958343506, -0.27236491441726685, 0.2715114951133728, 0.26292920112609863, -0.1263890266418457, -0.37431198358535767, -0.2648187279701233, 0.22566699981689453, 0.06889081001281738], dtype='float32').reshape([20]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([20]),
        ]


class PrimitiveOp_5de5dca7af4528b013606c41f9180211(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f59e29cf1d3df87426ffbc36f73a1fab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.32311898469924927], [-0.4162471294403076], [-0.24600857496261597], [-0.4810544550418854], [0.38517266511917114]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.30723488330841064], [0.35330432653427124], [-0.01109778881072998], [0.1941872239112854], [-0.17401427030563354]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_6c88b8ac8245e3ab261a4c0618c5fb74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.1121317446231842], [0.17175018787384033], [-0.021406471729278564], [-0.28133857250213623], [-0.06455051898956299]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.24333494901657104], [0.034552812576293945], [0.33392423391342163], [-0.12716543674468994], [0.4143192768096924]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_d6201b04f0cc2d4a83c05d9c960e336b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2732882499694824], [0.21844327449798584], [0.1365358829498291], [0.1824830174446106], [0.38517266511917114]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.11598128080368042], [-0.19975826144218445], [-0.01109778881072998], [0.1941872239112854], [-0.17401427030563354]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_8c0411b84bb773aeac9cd54652908c1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.47203218936920166], [0.33248358964920044], [-0.021406471729278564], [-0.28133857250213623], [0.33753883838653564]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.49989163875579834], [-0.10271179676055908], [0.22155678272247314], [-0.12957623600959778], [0.27495449781417847]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_9c5eb89c45346347e38c97fd86181599(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.32311898469924927], [-0.4162471294403076], [-0.24600857496261597], [-0.4810544550418854], [0.46119225025177]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.30723488330841064], [0.35330432653427124], [-0.4845125675201416], [-0.1398959755897522], [-0.31891727447509766]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_46db5554c5ea7cc6bb4232aaeb32e355(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.1121317446231842], [0.17175018787384033], [0.2869950532913208], [-0.10179561376571655], [-0.06455051898956299]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.24333494901657104], [0.034552812576293945], [0.33392423391342163], [-0.12716543674468994], [0.4143192768096924]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_8a71d28d65e123a0083614cf3db069c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.4610447883605957], [0.07641894370317459], [-0.04706235229969025], [-0.006878872402012348], [-0.33857452869415283]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_121b92066175b79b651d43bc9470aa75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2732882499694824], [0.21844327449798584], [0.1365358829498291], [0.1824830174446106], [0.46119225025177]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.11598128080368042], [-0.19975826144218445], [-0.4845125675201416], [-0.1398959755897522], [-0.31891727447509766]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_d77de408a539a325a3ff3835dd2097f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.47203218936920166], [0.33248358964920044], [0.2869950532913208], [-0.10179561376571655], [0.33753883838653564]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.49989163875579834], [-0.10271179676055908], [0.22155678272247314], [-0.12957623600959778], [0.27495449781417847]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_b039f4692f7e8392241958e05820cbe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.37834033370018005], [0.18199938535690308], [0.04064033553004265], [0.008955889381468296], [0.048822641372680664]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.4610447883605957], [0.07641894370317459], [-0.04706235229969025], [-0.006878872402012348], [-0.33857452869415283]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_c27bb724665bc18e804c40b1b8fae4ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [0.0], [-0.0], [-0.0], [-0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.21859803795814514], [0.5801143050193787], [2.1580207347869873], [1.7680836915969849], [7.93478536605835]], dtype='float32').reshape([5, 1]),
        ]


class PrimitiveOp_af2a3e46aa6086c6eee7787fcc949f21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 65536, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 65536, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf5a3c006d3706727d420972b674c7e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af2a3e46aa6086c6eee7787fcc949f21
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 65536, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_89c29ef55573b5d13699b760459cfc64(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[15200, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_822027155acd7ed8896200a3113b67cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89c29ef55573b5d13699b760459cfc64
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
            paddle.uniform([15200, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_822027155acd7ed8896200a3113b67cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89c29ef55573b5d13699b760459cfc64
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
            paddle.uniform([15200, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_410843445982599c787c12b8e449b47d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1830, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a8bdf7f8cee229573c2b68d0859ea601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a8bdf7f8cee229573c2b68d0859ea601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a8bdf7f8cee229573c2b68d0859ea601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a8bdf7f8cee229573c2b68d0859ea601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a8bdf7f8cee229573c2b68d0859ea601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a8bdf7f8cee229573c2b68d0859ea601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a8bdf7f8cee229573c2b68d0859ea601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a8bdf7f8cee229573c2b68d0859ea601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a8bdf7f8cee229573c2b68d0859ea601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a8bdf7f8cee229573c2b68d0859ea601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a8bdf7f8cee229573c2b68d0859ea601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3a4c83f203e9f67b723bc396bd4151be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3549, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3549, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b4038219c2cdc9919f3dcddb0dbd6e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a4c83f203e9f67b723bc396bd4151be
    def get_inputs(self):
        return [
            paddle.uniform([3549, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c7e07e42ba4884aed6c7a3a1e4ff1f18(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[3549, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_638ea59ba2f3e1c54ba4a51816db65e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7e07e42ba4884aed6c7a3a1e4ff1f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3549, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_410843445982599c787c12b8e449b47d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1830, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0bf4b4cbf5ed9a0d22464e3c0dc12397(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a2266005438d7ab42e3ede14dadaa410(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bf4b4cbf5ed9a0d22464e3c0dc12397
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e05fe3cf2467de3c306430df7fbf6b1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8, 512, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f4977291377e2444371f761ba6faf0bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e05fe3cf2467de3c306430df7fbf6b1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0368fad43623b73e790d949bddeb3e4a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            paddle.static.InputSpec(shape=[16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6013d9e688618c9fc50782a4f6322910(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0368fad43623b73e790d949bddeb3e4a
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
            paddle.to_tensor([0.47761791944503784, -0.13097253441810608, 0.18176352977752686, -0.38882213830947876, 0.041665852069854736, 0.32335561513900757, 0.38663744926452637, -0.4313998818397522, -0.1984555423259735, -0.3228505253791809, -0.10493266582489014, -0.2662540078163147, -0.3995301127433777, -0.019113868474960327, 0.3397362232208252, 0.11273664236068726], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_c04e682bdd8bf67005eecbf30f0f1b61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0368fad43623b73e790d949bddeb3e4a
    def get_inputs(self):
        return [
            paddle.to_tensor([0.47761791944503784, -0.13097253441810608, 0.18176352977752686, -0.38882213830947876, 0.041665852069854736, 0.32335561513900757, 0.38663744926452637, -0.4313998818397522, -0.1984555423259735, -0.3228505253791809, -0.10493266582489014, -0.2662540078163147, -0.3995301127433777, -0.019113868474960327, 0.3397362232208252, 0.11273664236068726], dtype='float32').reshape([16]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([16]),
        ]


class PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_538a6d9dddbb17fb367bbd1851c2b662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.uniform([4850], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4850], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_538a6d9dddbb17fb367bbd1851c2b662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.uniform([4850], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4850], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f0404045439c4a8d0a5ebe1ec095e9c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 2304, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 2304, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa01c79d6754a40da88e6dfbe140b3cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0404045439c4a8d0a5ebe1ec095e9c8
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 2304, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b4f6d05bff0a3035480da63593bc76bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21760, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 21760, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f35ff9b63687fe0f4fd6752c1264db7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4f6d05bff0a3035480da63593bc76bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 21760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d36aec7c70a56c55812de3adf0565e45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 198, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[54, 198, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3abd04a3e23517bd644a02218b4f8c8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d36aec7c70a56c55812de3adf0565e45
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 198, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_63ca1262f064d71ae996fd8ccbd37cb3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 3, 198, 198], dtype='float32'),
            paddle.static.InputSpec(shape=[54, 3, 198, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e19f67a60dd354a68efdc83a84bafef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63ca1262f064d71ae996fd8ccbd37cb3
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 198], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 3, 198, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c33fb23b7ea48e6e6cb687eafdcf3a6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([3039, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c1e1c96688331a49e5d530064f51a229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c1e1c96688331a49e5d530064f51a229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c1e1c96688331a49e5d530064f51a229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c1e1c96688331a49e5d530064f51a229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c1e1c96688331a49e5d530064f51a229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c1e1c96688331a49e5d530064f51a229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c1e1c96688331a49e5d530064f51a229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c1e1c96688331a49e5d530064f51a229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c1e1c96688331a49e5d530064f51a229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c1e1c96688331a49e5d530064f51a229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c1e1c96688331a49e5d530064f51a229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d2337ffc18b23e5e45ccf0feb0356b33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6069, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6069, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf74d7b1c0bb61ce62c2e2f974170249(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2337ffc18b23e5e45ccf0feb0356b33
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6069, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6eeea1fc78dff7dc1bbecfeb8f6a86ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[6069, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_47977443ec777bb9727f90766ec9947f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6eeea1fc78dff7dc1bbecfeb8f6a86ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6069, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c33fb23b7ea48e6e6cb687eafdcf3a6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([3039, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7cad50791374cc1aa03b5e99df27d987(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9772b7e697991b015fa042de52cd5f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9772b7e697991b015fa042de52cd5f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9772b7e697991b015fa042de52cd5f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9772b7e697991b015fa042de52cd5f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9772b7e697991b015fa042de52cd5f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9772b7e697991b015fa042de52cd5f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9772b7e697991b015fa042de52cd5f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5fc41b784aaef09d5c8e83652e9f104d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3549, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2cda77d053924d3bf0873269235335ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fc41b784aaef09d5c8e83652e9f104d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3549, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6fe75c6de59c37f50a7d18e31e4ce992(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1024, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2401eb32a07595dfdfef0a05c28c789(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fe75c6de59c37f50a7d18e31e4ce992
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1024, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7749b6386f3d44d6dbf21fc5378de926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab7278de0bfd314eb4d8dfd8fc87222c
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 1024, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8f894fb8191dda138ef80fe732fac8c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9935ad00e463f22cce5a3d36607f6b
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8b9a542951a1f429d6562b6eb793ca37(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d32428ea16117d87002afad85bbd06cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b9a542951a1f429d6562b6eb793ca37
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5bb0e222865ae94b855382033ed43e1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_79379dc10f62cacdc2db061aeb22d401(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb0e222865ae94b855382033ed43e1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_44d53373fadf4732b0f93853f02c9d5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6eff661b5176970610103901c79f9b1
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c744f130b5d4766b87a4c332e654b886(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8816, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[8816, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a285f71f0ed4bee33791f5d24dd316c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c744f130b5d4766b87a4c332e654b886
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8899f76134e74e4bc40d1814f72eedaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfa3b4dcb0b3a0e55b5821ad7c27cbbb
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a3f54a52ed23dfce1c197b502bc2a6cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_98a0b0a6637f351a58ca1c4e65492266(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3f54a52ed23dfce1c197b502bc2a6cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d65965f879918926c79768cf3a2702ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_81606fcb49e81c90ae664aa9fde7d13b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d65965f879918926c79768cf3a2702ba
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fecb3f5ff91d6aa7ee171e3042b0ebc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.26340150833129883], [-0.0067379772663116455], [-0.3218809962272644], [-0.08197793364524841], [-0.4799332618713379], [-0.08537876605987549], [-0.052667051553726196], [-0.4440675973892212], [-0.432447612285614]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.4368530511856079], [0.46684616804122925], [0.23317939043045044], [0.343126118183136], [0.4215221405029297], [0.45149946212768555], [-0.2904985547065735], [0.3288099765777588], [0.3321917653083801]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_5800fdbd148a8fab793d97990b048c3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.009119778871536255], [0.31540924310684204], [-0.2665252685546875], [-0.12849152088165283], [-0.11918747425079346], [-0.4604017734527588], [-0.4158199727535248], [-0.48267507553100586], [-0.3419676125049591]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.10920676589012146], [0.2152063250541687], [0.19116997718811035], [0.1771603226661682], [0.1994050145149231], [0.40892404317855835], [0.4096217751502991], [-0.0013248622417449951], [-0.025652199983596802]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_7d5c376326d545c4fefa229879d1995c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.26340150833129883], [0.46034830808639526], [0.16207528114318848], [-0.08197793364524841], [-0.4799332618713379], [-0.08537876605987549], [0.2270095944404602], [-0.21020689606666565], [-0.432447612285614]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.4368530511856079], [0.46684616804122925], [0.23317939043045044], [0.343126118183136], [0.37048572301864624], [0.12745672464370728], [-0.2904985547065735], [0.3288099765777588], [0.3160330057144165]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_2c1beaa676356ab1450ae71e08c9fd3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07887744903564453], [0.31540924310684204], [-0.2665252685546875], [-0.12849152088165283], [0.4635170102119446], [-0.1288277506828308], [-0.4158199727535248], [0.08123522996902466], [-0.3419676125049591]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.3653436005115509], [-0.4201619625091553], [-0.20712703466415405], [0.1771603226661682], [-0.06482309103012085], [0.40892404317855835], [0.18776559829711914], [-0.0013248622417449951], [-0.025652199983596802]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_aeaae044aef63c702081c27d7e79f509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.30383044481277466], [-0.0067379772663116455], [-0.3218809962272644], [0.3049466609954834], [0.342303991317749], [0.4855678081512451], [-0.052667051553726196], [-0.4440675973892212], [0.3661329746246338]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.2951025366783142], [0.1575915813446045], [0.18225038051605225], [0.010741472244262695], [0.4215221405029297], [0.45149946212768555], [-0.3987027704715729], [-0.3669936954975128], [0.3321917653083801]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_b44f58ebbeb7332b96f70fe0e017f86a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.009119778871536255], [0.35780632495880127], [-0.07009202241897583], [0.13681113719940186], [-0.11918747425079346], [-0.4604017734527588], [0.17320948839187622], [-0.48267507553100586], [-0.1678488850593567]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.10920676589012146], [0.2152063250541687], [0.19116997718811035], [0.02454829216003418], [0.1994050145149231], [-0.48527729511260986], [0.4096217751502991], [-0.12472957372665405], [-0.4282859265804291]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_9b6a348f694347f7dfb89ceb5a39e5a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.07617727667093277], [-0.028213033452630043], [0.13593381643295288], [0.1629621386528015], [-0.42407214641571045], [0.11530013382434845], [-0.39416757225990295], [-0.016913022845983505], [0.24559549987316132]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_f137fe68acc5d73225c4589ef0b32cdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.30383044481277466], [0.46034830808639526], [0.16207528114318848], [0.3049466609954834], [0.342303991317749], [0.4855678081512451], [0.2270095944404602], [-0.21020689606666565], [0.3661329746246338]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.2951025366783142], [0.1575915813446045], [0.18225038051605225], [0.010741472244262695], [0.37048572301864624], [0.12745672464370728], [-0.3987027704715729], [-0.3669936954975128], [0.3160330057144165]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_4ac74dede4763b7381787a691ffdbb93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07887744903564453], [0.35780632495880127], [-0.07009202241897583], [0.13681113719940186], [0.4635170102119446], [-0.1288277506828308], [0.17320948839187622], [0.08123522996902466], [-0.1678488850593567]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.3653436005115509], [-0.4201619625091553], [-0.20712703466415405], [0.02454829216003418], [-0.06482309103012085], [-0.48527729511260986], [0.18776559829711914], [-0.12472957372665405], [-0.4282859265804291]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_94a3e77851c11b36869754e888323678(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0038771205581724644], [0.235535129904747], [-0.002764695091173053], [0.033028312027454376], [-0.01488953921943903], [0.12764853239059448], [-0.009107938036322594], [0.03229256346821785], [0.013047887943685055]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.07617727667093277], [-0.028213033452630043], [0.13593381643295288], [0.1629621386528015], [-0.42407214641571045], [0.11530013382434845], [-0.39416757225990295], [-0.016913022845983505], [0.24559549987316132]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_eb71696d912f8036891904125bc0b4d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [-0.0], [0.0], [0.0], [-0.0], [0.0], [-0.0], [-0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[20.647899627685547], [1.1197826862335205], [50.16774368286133], [-3.934013843536377], [-27.48121452331543], [0.09673748910427094], [-42.277366638183594], [1.5237436294555664], [-17.82262420654297]], dtype='float32').reshape([9, 1]),
        ]


class PrimitiveOp_429e2045e6f4da7e01c19b5d8020bfd2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 19, 32768], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 19, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6d3e4d6b1a64bf1309b34f817507cf33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_429e2045e6f4da7e01c19b5d8020bfd2
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 32768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[0.49999988079071045], [0.4999616742134094], [0.49995607137680054], [0.4999149441719055], [0.49995774030685425], [0.49986618757247925], [0.4999755024909973], [0.49996358156204224], [0.499997615814209], [0.4999924302101135], [0.49994170665740967], [0.4999611973762512], [0.4999982714653015], [0.49995172023773193], [0.4999915361404419], [0.49999290704727173], [0.4999861717224121], [0.499975323677063], [0.4999634027481079]]], dtype='float32').reshape([1, 19, 1]),
        ]


class TestPrimitiveOp_95d47ed588bcea580d7add676d09ba08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_080a81e4a7b89445069bd25d1db2cd26
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 9216, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_826bcb0c5a3a26873f57ff6060900348(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d1ead9d2e5795eead51e28290f48880(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_826bcb0c5a3a26873f57ff6060900348
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cbe827cea03aaca01eb121df603bc8ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1be0a4b403156666fb0723cfe10b1d49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbe827cea03aaca01eb121df603bc8ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_910711994f9d5ba4e6c272958d1870e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 49, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6bb42f504959ff667b2e8161341ac358(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_910711994f9d5ba4e6c272958d1870e8
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e8143c15f04e061ebc91a274e6e87c98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dfcedf316eacbbadf7dd6cacaded896
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.016451716423034668]], [[0.19434791803359985]], [[0.13872134685516357]], [[0.23950600624084473]], [[-0.19793948531150818]], [[0.47337377071380615]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6401166319847107]], [[0.354726105928421]], [[0.31728193163871765]], [[0.5016976594924927]], [[0.3737022876739502]], [[0.3746196925640106]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_1f5c723708f448025f3523f9c2939b80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dfcedf316eacbbadf7dd6cacaded896
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.17284101247787476]], [[-0.13571462035179138]], [[0.010938167572021484]], [[0.11076635122299194]], [[0.29814577102661133]], [[-0.3025370240211487]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.7830853462219238]], [[0.32650431990623474]], [[0.4100632071495056]], [[0.5074648857116699]], [[0.7702154517173767]], [[0.6741681694984436]]], dtype='float32').reshape([6, 1, 1]),
        ]


class PrimitiveOp_43892ef43009d9b41be8211d72b6e90d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 50, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 50, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c391ac31e77be78b3a1f2ce1275ab62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43892ef43009d9b41be8211d72b6e90d
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 50, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_809fdcfe05bb9dbf3dd2db42991ace51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cea6fb04c6ab3a7c9d9cf93206b2bb68
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3b2df143cb29ff6143b6ee5928593087(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2046, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ed3e4dd75c4d66439c0e57c307bde76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ed3e4dd75c4d66439c0e57c307bde76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ed3e4dd75c4d66439c0e57c307bde76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ed3e4dd75c4d66439c0e57c307bde76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ed3e4dd75c4d66439c0e57c307bde76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ed3e4dd75c4d66439c0e57c307bde76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ed3e4dd75c4d66439c0e57c307bde76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ed3e4dd75c4d66439c0e57c307bde76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ed3e4dd75c4d66439c0e57c307bde76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ed3e4dd75c4d66439c0e57c307bde76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ed3e4dd75c4d66439c0e57c307bde76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_99d2d5c0234b7757356679e379bcde16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50398eaf8f3d9b9c285466c003f7bb78
    def get_inputs(self):
        return [
            paddle.uniform([4116, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4116, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b00b3320963b03a2a16c4a8be000d2f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0808c9fcbc960a398d51addf53e19cab
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4116, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3b2df143cb29ff6143b6ee5928593087(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2046, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_539c973cda41b214f69f861db982f127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_539c973cda41b214f69f861db982f127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_539c973cda41b214f69f861db982f127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_539c973cda41b214f69f861db982f127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_539c973cda41b214f69f861db982f127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_539c973cda41b214f69f861db982f127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_539c973cda41b214f69f861db982f127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_539c973cda41b214f69f861db982f127(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9544a1dc96ec4c05e13a86968e8c439d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9544a1dc96ec4c05e13a86968e8c439d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9544a1dc96ec4c05e13a86968e8c439d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9544a1dc96ec4c05e13a86968e8c439d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9544a1dc96ec4c05e13a86968e8c439d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9544a1dc96ec4c05e13a86968e8c439d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9544a1dc96ec4c05e13a86968e8c439d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9544a1dc96ec4c05e13a86968e8c439d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_feea3775d2244a236caefac5af9726fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_adc0d5b94a3ba5b6ea922ae95808dfd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_adc0d5b94a3ba5b6ea922ae95808dfd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_adc0d5b94a3ba5b6ea922ae95808dfd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_adc0d5b94a3ba5b6ea922ae95808dfd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_adc0d5b94a3ba5b6ea922ae95808dfd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_adc0d5b94a3ba5b6ea922ae95808dfd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_adc0d5b94a3ba5b6ea922ae95808dfd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6cc9af3a989f9c0621dd0fcb73d9a86b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[150, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[150, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69b2ed63abd121a63b6a2c2494a1f4da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cc9af3a989f9c0621dd0fcb73d9a86b
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3b2504e6c19ad4850aa8afad3437f488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_733c155888a4588c13781126b76103a8
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_09add4a452a351a1e1c55f0ac0e9c64d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1174, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5bac775167500a129018512c5843bcbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09add4a452a351a1e1c55f0ac0e9c64d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1174, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_411655d251b21b2ba09b836e753362a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 1174, 1174], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 12, 1174, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_271eec11d885ad3d2e0f844c623d88e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_411655d251b21b2ba09b836e753362a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 1174, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a0e2704049bf3916cc98dd2e3b3fb6c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4725, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c9fd742f9231898eda8e82cfcbf2075a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0e2704049bf3916cc98dd2e3b3fb6c5
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4725, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_430cc7b72b1543d446e4ac9fb9a38353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_430cc7b72b1543d446e4ac9fb9a38353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_430cc7b72b1543d446e4ac9fb9a38353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_430cc7b72b1543d446e4ac9fb9a38353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_430cc7b72b1543d446e4ac9fb9a38353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_430cc7b72b1543d446e4ac9fb9a38353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_430cc7b72b1543d446e4ac9fb9a38353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d085ea8850d092065ac263e9414a4294(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 11109, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b3b379f9aa1b91ddd1f1ac9f16447f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d085ea8850d092065ac263e9414a4294
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 11109, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4df3e794288c9c5e55f7a3e19899ac69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 198, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[86, 198, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49552cd33b7f82f246fe91cc90f817a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4df3e794288c9c5e55f7a3e19899ac69
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 198, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_405d34c42a0945d3481aa384f2cd4b1d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 198, 198], dtype='float32'),
            paddle.static.InputSpec(shape=[86, 3, 198, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_185c80bf6f83f80822224f1b4d6d29e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_405d34c42a0945d3481aa384f2cd4b1d
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 198], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 3, 198, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8b2c1714ff69c46636eb8bd6c69c7735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ee85a9a59cd15f4a297868d77aa7453
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7ec861ae84f6c4868f97ea1102e0a100(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[70, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[70, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a518550062812db5096307b7e885a92a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ec861ae84f6c4868f97ea1102e0a100
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1c8fb9abf758462310e4fa6135718407(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 9216, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 9216, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b7fa210327a295cf75074f007ac0643(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c8fb9abf758462310e4fa6135718407
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 9216, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fe63cab98da7d8138e0d1d86917b2864(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12096, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 12096, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d0364fae5a0818692c8c4bd771470468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe63cab98da7d8138e0d1d86917b2864
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12096, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e357b570657f6fab550c03fc9bba497e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1aa4da7d3bae3db22c01fe22de924c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1aa4da7d3bae3db22c01fe22de924c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1aa4da7d3bae3db22c01fe22de924c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1aa4da7d3bae3db22c01fe22de924c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1aa4da7d3bae3db22c01fe22de924c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1aa4da7d3bae3db22c01fe22de924c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1aa4da7d3bae3db22c01fe22de924c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1aa4da7d3bae3db22c01fe22de924c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5dabcdad19e17062c4224151836e2997(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_197a7e805cb368257c02d15ff9bf5bee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_197a7e805cb368257c02d15ff9bf5bee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_197a7e805cb368257c02d15ff9bf5bee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_197a7e805cb368257c02d15ff9bf5bee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_197a7e805cb368257c02d15ff9bf5bee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_197a7e805cb368257c02d15ff9bf5bee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_197a7e805cb368257c02d15ff9bf5bee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_197a7e805cb368257c02d15ff9bf5bee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7c71d0613eb76a3ad28beed4f057d114(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8192, 8192], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8192, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57fb9b6c5a1297864510a97561979410(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c71d0613eb76a3ad28beed4f057d114
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 8192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8192, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_73583fca9150233dc45d91eabd225d9c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_29f033b019b7685e7d891bbf4e758af4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73583fca9150233dc45d91eabd225d9c
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_990ba952656ccc8bfa777ab24221b131(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.49611642956733704]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.23364883661270142]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_423e0316cab181652d98cd96a347297c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.13599738478660583]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.10313171148300171]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_96798c6aa7a04bba4d2df9c82dada30e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.009030580520629883]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.4661974608898163]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_ae3e2bfce72ccf776fbb2db52b5d67f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17337018251419067]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.10313171148300171]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_990ba952656ccc8bfa777ab24221b131(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.49611642956733704]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.23364883661270142]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_6d84b4ef061381d9dcdeb5e730945112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.13599738478660583]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.22258034348487854]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_1ca8c8791fd13e4c2dd6dc507b43cdbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.029805947095155716]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_96798c6aa7a04bba4d2df9c82dada30e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.009030580520629883]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.4661974608898163]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_d3673b580814d42da1ecfadd6a2ea154(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17337018251419067]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.22258034348487854]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_b0534070dea29e2fc33aa01df8609677(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1881667971611023]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.029805947095155716]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_e3c894fdc93fce3788a0dbdbc11a8a4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[1.1584017276763916]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_56d6f7b95c20539f60e6e01de098bffc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52d5008448878c67a6c2e468a48568d3
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_113f291148f8bc32b7236b768ae44a90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4367c13caf8c619ef448818b71b78e9
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1c391ac31e77be78b3a1f2ce1275ab62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43892ef43009d9b41be8211d72b6e90d
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 50, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7f06dae19e48e1b8efdf372673ea13d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 50, 50], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 8, 50, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ac07b73fad54bc76ca147355912cc12c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f06dae19e48e1b8efdf372673ea13d6
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 8, 50, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a3fabb513d00823b46cfe60432a80317(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6069, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d88633c21bf165095d61fb6e03ab303a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3fabb513d00823b46cfe60432a80317
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6069, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3bbd40843ed8df6da0b1fa7f67111020(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[56, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4f976d8f7ed5ba378176fc895d858778(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bbd40843ed8df6da0b1fa7f67111020
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([56, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_197a7e805cb368257c02d15ff9bf5bee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_197a7e805cb368257c02d15ff9bf5bee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_197a7e805cb368257c02d15ff9bf5bee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_197a7e805cb368257c02d15ff9bf5bee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_197a7e805cb368257c02d15ff9bf5bee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_197a7e805cb368257c02d15ff9bf5bee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_197a7e805cb368257c02d15ff9bf5bee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3b2504e6c19ad4850aa8afad3437f488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_733c155888a4588c13781126b76103a8
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ec4e75368cf214afccd0c43a4f10be16(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 576, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 576, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cbe309cc939a8cc8fce773576c954cd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec4e75368cf214afccd0c43a4f10be16
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 576, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_209c9c6f5944874ca063ae769571805a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1025, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f721ee95ecb70d6b7618bb6171f3729(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_209c9c6f5944874ca063ae769571805a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1025, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_50068db0e334e45a4722ab6ec9501e3a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1025, 1025], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6, 1025, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa48a3a7eb8a9ac02a00253dc7eed1cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50068db0e334e45a4722ab6ec9501e3a
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6, 1025, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9921b595bb1852afb4a7eb1e293d3f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9921b595bb1852afb4a7eb1e293d3f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9921b595bb1852afb4a7eb1e293d3f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9921b595bb1852afb4a7eb1e293d3f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9921b595bb1852afb4a7eb1e293d3f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9921b595bb1852afb4a7eb1e293d3f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9921b595bb1852afb4a7eb1e293d3f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_721630adc70c95896527591cc9fdbb05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_721630adc70c95896527591cc9fdbb05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_721630adc70c95896527591cc9fdbb05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_721630adc70c95896527591cc9fdbb05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_721630adc70c95896527591cc9fdbb05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_721630adc70c95896527591cc9fdbb05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_721630adc70c95896527591cc9fdbb05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_444fd054beaf6f5a804d00e717d6bee9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([5498, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef48c0c2713b03965fedaf6bf0061ff5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef48c0c2713b03965fedaf6bf0061ff5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef48c0c2713b03965fedaf6bf0061ff5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef48c0c2713b03965fedaf6bf0061ff5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef48c0c2713b03965fedaf6bf0061ff5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef48c0c2713b03965fedaf6bf0061ff5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef48c0c2713b03965fedaf6bf0061ff5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef48c0c2713b03965fedaf6bf0061ff5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef48c0c2713b03965fedaf6bf0061ff5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef48c0c2713b03965fedaf6bf0061ff5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef48c0c2713b03965fedaf6bf0061ff5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6c7f912849971489ee33e3c279221b79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11109, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 11109, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6806f77cfae2c911223cf8cc80180bb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c7f912849971489ee33e3c279221b79
    def get_inputs(self):
        return [
            paddle.uniform([11109, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 11109, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f5eb5f4b24b083f85bc38eeaaf2951db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[11109, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c3d8dbfb0b82976781908de964ba741(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5eb5f4b24b083f85bc38eeaaf2951db
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11109, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_444fd054beaf6f5a804d00e717d6bee9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([5498, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b204dfed144dea58ca52a1f7e27ade54(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8400, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cde41d2a7230c4daa0c4aa602a7ea099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b204dfed144dea58ca52a1f7e27ade54
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8400, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9a1123d79106ceb97abd981d8a72507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9a1123d79106ceb97abd981d8a72507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9a1123d79106ceb97abd981d8a72507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9a1123d79106ceb97abd981d8a72507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9a1123d79106ceb97abd981d8a72507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9a1123d79106ceb97abd981d8a72507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9a1123d79106ceb97abd981d8a72507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9a1123d79106ceb97abd981d8a72507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1aa4da7d3bae3db22c01fe22de924c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1aa4da7d3bae3db22c01fe22de924c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1aa4da7d3bae3db22c01fe22de924c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1aa4da7d3bae3db22c01fe22de924c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1aa4da7d3bae3db22c01fe22de924c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1aa4da7d3bae3db22c01fe22de924c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1aa4da7d3bae3db22c01fe22de924c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88f1147c4c5a942697f55aa90a1410a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1074, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_002ec0477c84e42b118864db3b43cf44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_002ec0477c84e42b118864db3b43cf44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_002ec0477c84e42b118864db3b43cf44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_002ec0477c84e42b118864db3b43cf44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_002ec0477c84e42b118864db3b43cf44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_002ec0477c84e42b118864db3b43cf44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_002ec0477c84e42b118864db3b43cf44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_002ec0477c84e42b118864db3b43cf44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_002ec0477c84e42b118864db3b43cf44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_002ec0477c84e42b118864db3b43cf44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_002ec0477c84e42b118864db3b43cf44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0386906bae6ceafa65c95a5bb420c828(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2100, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2100, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3c8f2a0549909180dbbb3b7a4dd26381(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0386906bae6ceafa65c95a5bb420c828
    def get_inputs(self):
        return [
            paddle.uniform([2100, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2100, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_639ea61878bad07af55eb84bbc394179(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[2100, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a9c4880994141e0afae5c95d00af9ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_639ea61878bad07af55eb84bbc394179
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2100, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88f1147c4c5a942697f55aa90a1410a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1074, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_098ae969f497f6d23855ee3a64346eab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[53, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[53, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3950bcc465616e38b9954afb0fcdd012(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_098ae969f497f6d23855ee3a64346eab
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([53, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6bb42f504959ff667b2e8161341ac358(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_910711994f9d5ba4e6c272958d1870e8
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_414db5110a325f7d2a6ccb7b2d753bd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1773, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d72bbcf30896eb03b9548489ab0a9d57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d72bbcf30896eb03b9548489ab0a9d57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d72bbcf30896eb03b9548489ab0a9d57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d72bbcf30896eb03b9548489ab0a9d57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d72bbcf30896eb03b9548489ab0a9d57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d72bbcf30896eb03b9548489ab0a9d57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d72bbcf30896eb03b9548489ab0a9d57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d72bbcf30896eb03b9548489ab0a9d57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d72bbcf30896eb03b9548489ab0a9d57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d72bbcf30896eb03b9548489ab0a9d57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d72bbcf30896eb03b9548489ab0a9d57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b4038219c2cdc9919f3dcddb0dbd6e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a4c83f203e9f67b723bc396bd4151be
    def get_inputs(self):
        return [
            paddle.uniform([3549, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_638ea59ba2f3e1c54ba4a51816db65e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7e07e42ba4884aed6c7a3a1e4ff1f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3549, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_414db5110a325f7d2a6ccb7b2d753bd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1773, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5095cc193929debdbfb4f6eabc80c45a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 197, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[86, 197, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55c7f7ef2bdcaa9804a54525dc338cd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5095cc193929debdbfb4f6eabc80c45a
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 197, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ad16d7e51b98ae4484ec9921963e2231(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 3, 197, 197], dtype='float32'),
            paddle.static.InputSpec(shape=[86, 3, 197, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc600a4d4095b70e0e149ef439b9e83f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad16d7e51b98ae4484ec9921963e2231
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 197], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 3, 197, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_126c3c3d01850f8f067cc2c80e4f96d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d642141b66a724f45dae974c1fd5dce
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_206da4c03be6a988c08e2f59f6306763(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[84, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[84, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d2bbeb5f968c788bdb30343032ffb597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_206da4c03be6a988c08e2f59f6306763
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_77f4cca422d2eb95a42255bdc8fd32bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4, 1, 13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b05eb2281b063d3792fca03b46f66d3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f4cca422d2eb95a42255bdc8fd32bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4, 1, 13, 19], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3bd71b1091a034820703724d3bddf585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3bd71b1091a034820703724d3bddf585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3bd71b1091a034820703724d3bddf585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3bd71b1091a034820703724d3bddf585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3bd71b1091a034820703724d3bddf585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3bd71b1091a034820703724d3bddf585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3bd71b1091a034820703724d3bddf585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3bd71b1091a034820703724d3bddf585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b0df867a4644c834ec9fd270790bdf2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            paddle.static.InputSpec(shape=[24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_edd32c42843b4f04cd4ca89d0b945d36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0df867a4644c834ec9fd270790bdf2c
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
            paddle.to_tensor([-0.4896465539932251, -0.16032052040100098, 0.4041510820388794, -0.27794021368026733, -0.13702961802482605, 0.04065430164337158, 0.08473557233810425, -0.11503106355667114, 0.18012410402297974, 0.227092444896698, -0.05352669954299927, 0.0845482349395752, 0.2990933060646057, 0.4492619037628174, 0.4188406467437744, -0.3856993019580841, -0.4507875144481659, -0.030939042568206787, 0.0364069938659668, 0.07344573736190796, 0.39202117919921875, 0.06445503234863281, 0.01184624433517456, -0.358212947845459], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_f7eb14001324307df4f9ecf8794a637c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0df867a4644c834ec9fd270790bdf2c
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.4896465539932251, -0.16032052040100098, 0.4041510820388794, -0.27794021368026733, -0.13702961802482605, 0.04065430164337158, 0.08473557233810425, -0.11503106355667114, 0.18012410402297974, 0.227092444896698, -0.05352669954299927, 0.0845482349395752, 0.2990933060646057, 0.4492619037628174, 0.4188406467437744, -0.3856993019580841, -0.4507875144481659, -0.030939042568206787, 0.0364069938659668, 0.07344573736190796, 0.39202117919921875, 0.06445503234863281, 0.01184624433517456, -0.358212947845459], dtype='float32').reshape([24]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([24]),
        ]


class PrimitiveOp_9a27b98024aeb7b7964f1eea0679cc4f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4096, 4096], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4096, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_561c81736a63eb4b9706ea0b0fefe217(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a27b98024aeb7b7964f1eea0679cc4f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 4096], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4096, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7d7099444314e4123666d9648de3b91a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.uniform([17524], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([17524], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7d7099444314e4123666d9648de3b91a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.uniform([17524], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([17524], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3929f14f147410bb4e712b4b16c1521a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3929f14f147410bb4e712b4b16c1521a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3929f14f147410bb4e712b4b16c1521a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3929f14f147410bb4e712b4b16c1521a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3929f14f147410bb4e712b4b16c1521a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3929f14f147410bb4e712b4b16c1521a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3929f14f147410bb4e712b4b16c1521a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3929f14f147410bb4e712b4b16c1521a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bf160cb0afd58f872e2ab9eb81b52653(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[950, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_65072afa7dcef74ea287b50c742af815(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf160cb0afd58f872e2ab9eb81b52653
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
            paddle.uniform([950, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65072afa7dcef74ea287b50c742af815(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf160cb0afd58f872e2ab9eb81b52653
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
            paddle.uniform([950, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4e446c6282d49c7e42083ccf26c2e93d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 577, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 577, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23ea6e06676fb33a0fdf01db2586c100(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e446c6282d49c7e42083ccf26c2e93d
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 577, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d6c8c32aa58ce12003ed458ef58494e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 12, 577, 577], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 12, 577, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b4fd1d721f601306fbdcd38055dc5b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6c8c32aa58ce12003ed458ef58494e6
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 577, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3b2504e6c19ad4850aa8afad3437f488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_733c155888a4588c13781126b76103a8
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_35c5853473822729f000e399843e2039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30894a70778990ba804af814e417d17d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a1a87515d713804f0946dc2bc4c24a38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b38cea3f6f5c141b129a6d4ec2046a84
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c59e47067793c1a15e665eef9d0fa7a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 17, 25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4, 1, 25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b1e3e64f2f77434a22490a39395c4ca2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c59e47067793c1a15e665eef9d0fa7a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4, 1, 25, 38], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c48802e6fe1109a80857a5f232e4acef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9927cc6f9e052f94df4befead71a14f9
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1b7fa210327a295cf75074f007ac0643(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c8fb9abf758462310e4fa6135718407
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 9216, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_809fdcfe05bb9dbf3dd2db42991ace51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cea6fb04c6ab3a7c9d9cf93206b2bb68
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_af9b2c5ceda54843c3209c8ac38248ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e03f9187b8b7dac91f740b6d9ed74562
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 160, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2c37e7ac623f17f927ff795c90add074(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 8, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_87ce9ade8b4206af5f5de13fd8b215ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c37e7ac623f17f927ff795c90add074
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fb93aefb6749274dd2abf6d9e2772012(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 8, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 8, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f225a5781403609b7217bbe3c62a353c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb93aefb6749274dd2abf6d9e2772012
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b630987069d3aa1437c668cf0e1bc8e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16384, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[16384, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_88e98d3cbed44a33c04bc73c753cb302(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b630987069d3aa1437c668cf0e1bc8e0
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([16384, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_113f291148f8bc32b7236b768ae44a90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4367c13caf8c619ef448818b71b78e9
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0bf4846098a19b4a316fe88f8a4017c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0bf4846098a19b4a316fe88f8a4017c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0bf4846098a19b4a316fe88f8a4017c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0bf4846098a19b4a316fe88f8a4017c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0bf4846098a19b4a316fe88f8a4017c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0bf4846098a19b4a316fe88f8a4017c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0bf4846098a19b4a316fe88f8a4017c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9de42d9af379eb169b82ce1ed258d066(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[70, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5cbb33841172402b4b6f6929ee42ea1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9de42d9af379eb169b82ce1ed258d066
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
            paddle.uniform([70, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5cbb33841172402b4b6f6929ee42ea1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9de42d9af379eb169b82ce1ed258d066
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
            paddle.uniform([70, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2024a5ac1588a345bdffeca09745689f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 144, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 144, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_372088b7ee6bc7fb50425dfc416de172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2024a5ac1588a345bdffeca09745689f
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 144, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9772b7e697991b015fa042de52cd5f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9772b7e697991b015fa042de52cd5f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9772b7e697991b015fa042de52cd5f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9772b7e697991b015fa042de52cd5f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9772b7e697991b015fa042de52cd5f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9772b7e697991b015fa042de52cd5f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9772b7e697991b015fa042de52cd5f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9772b7e697991b015fa042de52cd5f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1cfd53e7ab5b4539c7c7dfcdc252c702(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a009f3831aa9518882502ffdcff086b
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_879d8dec58de1d2fbee7c93f856dbae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488780642555bbf420383e7fb6ba9d19
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aabb571e33c4a62dda3aa6ae2e3da4ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aabb571e33c4a62dda3aa6ae2e3da4ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aabb571e33c4a62dda3aa6ae2e3da4ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aabb571e33c4a62dda3aa6ae2e3da4ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aabb571e33c4a62dda3aa6ae2e3da4ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aabb571e33c4a62dda3aa6ae2e3da4ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aabb571e33c4a62dda3aa6ae2e3da4ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aabb571e33c4a62dda3aa6ae2e3da4ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_113f291148f8bc32b7236b768ae44a90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4367c13caf8c619ef448818b71b78e9
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f6d43a40fc7bcf1afed9ab2b55390f8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([4224, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_67dfbaf77f8f4c1a87bf9f4b2ebec169(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_67dfbaf77f8f4c1a87bf9f4b2ebec169(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_67dfbaf77f8f4c1a87bf9f4b2ebec169(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_67dfbaf77f8f4c1a87bf9f4b2ebec169(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_67dfbaf77f8f4c1a87bf9f4b2ebec169(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_67dfbaf77f8f4c1a87bf9f4b2ebec169(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_67dfbaf77f8f4c1a87bf9f4b2ebec169(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_67dfbaf77f8f4c1a87bf9f4b2ebec169(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_67dfbaf77f8f4c1a87bf9f4b2ebec169(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_67dfbaf77f8f4c1a87bf9f4b2ebec169(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_67dfbaf77f8f4c1a87bf9f4b2ebec169(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_810dd930926bb2280bc25735c998c35b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8400, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 8400, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80697583dffb98d9791c367ee1625848(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_810dd930926bb2280bc25735c998c35b
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8400, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_12441866cc2f113c2daef98e28c90533(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[8400, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df7468ebb62f3d64bde227f078786257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12441866cc2f113c2daef98e28c90533
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([8400, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f6d43a40fc7bcf1afed9ab2b55390f8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([4224, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e94d8f4bbad8687b70f1f5eb97ff700c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8e67cda4c004970d7e5da1ba47481fb
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_adc0d5b94a3ba5b6ea922ae95808dfd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_adc0d5b94a3ba5b6ea922ae95808dfd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_adc0d5b94a3ba5b6ea922ae95808dfd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_adc0d5b94a3ba5b6ea922ae95808dfd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_adc0d5b94a3ba5b6ea922ae95808dfd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_adc0d5b94a3ba5b6ea922ae95808dfd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_adc0d5b94a3ba5b6ea922ae95808dfd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_adc0d5b94a3ba5b6ea922ae95808dfd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5f89eb68374ad2098e7c331c62ca8046(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[551, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[551, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cee00d86cbd84ee1900b6415f311f421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f89eb68374ad2098e7c331c62ca8046
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d3ded76ac568482376e2cdb80714a674(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4096, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[4096, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_11731aadabf831935e8d0cb8a7292580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3ded76ac568482376e2cdb80714a674
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4096, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c48802e6fe1109a80857a5f232e4acef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9927cc6f9e052f94df4befead71a14f9
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6f7a6a36e6d451e1241a2a9a5d7441b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6804, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6804, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa4f49f31ed36471feeecda5fca1aec6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f7a6a36e6d451e1241a2a9a5d7441b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6804, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_86636cd620188e77aa5919cbf8b9d393(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[52, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[52, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6818fa503829add5323122011514d3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86636cd620188e77aa5919cbf8b9d393
    def get_inputs(self):
        return [
            paddle.uniform([52, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([52, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fabb9a4d545e157a51329090937d4782(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5376, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 5376, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e3918614aa77c2b15ccb5f87f2fd353b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fabb9a4d545e157a51329090937d4782
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 5376, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2fecccb1f5fe1f17fe22db17e70e8081(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 49, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7399e380d8fb9a091a9a1ab02c4cc18b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fecccb1f5fe1f17fe22db17e70e8081
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e4ae0016c375a776234e14b5a0109d20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[3800, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_483216cc8477a4947513420cd9f8def5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4ae0016c375a776234e14b5a0109d20
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
            paddle.uniform([3800, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_483216cc8477a4947513420cd9f8def5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4ae0016c375a776234e14b5a0109d20
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
            paddle.uniform([3800, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1cfd53e7ab5b4539c7c7dfcdc252c702(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a009f3831aa9518882502ffdcff086b
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_59b9440736e839d50791cbaf62b0eef4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1cdf7e55c209f160747e8650460062a3
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_29fca67531ae9c4d3049a6abf65a9ef4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 144, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 144, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_67ac5ab4e8e5edf80ef21a4eb5c3b417(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29fca67531ae9c4d3049a6abf65a9ef4
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 144, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2c17e42a237993ce122596447d6cc41e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([4657, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e208c637fe32274e1d6a72a26332e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e208c637fe32274e1d6a72a26332e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e208c637fe32274e1d6a72a26332e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e208c637fe32274e1d6a72a26332e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e208c637fe32274e1d6a72a26332e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e208c637fe32274e1d6a72a26332e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e208c637fe32274e1d6a72a26332e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e208c637fe32274e1d6a72a26332e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e208c637fe32274e1d6a72a26332e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e208c637fe32274e1d6a72a26332e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e208c637fe32274e1d6a72a26332e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4f39622dcf124217b362e6da2a03f1af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[9261, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 9261, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_51b4c9928582f83fcdc17a22b2a69a71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f39622dcf124217b362e6da2a03f1af
    def get_inputs(self):
        return [
            paddle.uniform([9261, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 9261, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a034f0558f586eb2751e2cf8dc120268(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[9261, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d094a26f90e0b9bab0676ec15e89fbff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a034f0558f586eb2751e2cf8dc120268
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([9261, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2c17e42a237993ce122596447d6cc41e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([4657, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a1a87515d713804f0946dc2bc4c24a38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b38cea3f6f5c141b129a6d4ec2046a84
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f4977291377e2444371f761ba6faf0bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e05fe3cf2467de3c306430df7fbf6b1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8f894fb8191dda138ef80fe732fac8c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9935ad00e463f22cce5a3d36607f6b
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7399e380d8fb9a091a9a1ab02c4cc18b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fecccb1f5fe1f17fe22db17e70e8081
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f37e49958f79b522a414351b5ab35068(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([3770, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_686dd4cd3c4247d558d2fcc552dc9d9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_686dd4cd3c4247d558d2fcc552dc9d9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_686dd4cd3c4247d558d2fcc552dc9d9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_686dd4cd3c4247d558d2fcc552dc9d9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_686dd4cd3c4247d558d2fcc552dc9d9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_686dd4cd3c4247d558d2fcc552dc9d9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_686dd4cd3c4247d558d2fcc552dc9d9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_686dd4cd3c4247d558d2fcc552dc9d9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_686dd4cd3c4247d558d2fcc552dc9d9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_686dd4cd3c4247d558d2fcc552dc9d9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_686dd4cd3c4247d558d2fcc552dc9d9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4319ca246764953320ce6e3f935bf6b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7581, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 7581, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a74f0609c6be5f92f831132c87f8c96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4319ca246764953320ce6e3f935bf6b5
    def get_inputs(self):
        return [
            paddle.uniform([7581, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 7581, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_33493f3e3759817ddee21e0107f59caf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[7581, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_086ab2480b4c1f01f78017be1f432229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33493f3e3759817ddee21e0107f59caf
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([7581, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f37e49958f79b522a414351b5ab35068(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([3770, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8899f76134e74e4bc40d1814f72eedaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfa3b4dcb0b3a0e55b5821ad7c27cbbb
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5d3448cde4f62d9499e9a52e8243e86d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a37b23951ff5101fe91834373112dd7
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 100, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_11297aae2f0c438bfee80a90f3f12d36(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1174, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_797615dc8a02ba111d6c2a3b8ab1166a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11297aae2f0c438bfee80a90f3f12d36
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1174, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ddbed025e010d7416f8df55ca9e32a67(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1174, 1174], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6, 1174, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a428bc020fda93b709972201e6480fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddbed025e010d7416f8df55ca9e32a67
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6, 1174, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_430cc7b72b1543d446e4ac9fb9a38353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_430cc7b72b1543d446e4ac9fb9a38353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_430cc7b72b1543d446e4ac9fb9a38353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_430cc7b72b1543d446e4ac9fb9a38353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_430cc7b72b1543d446e4ac9fb9a38353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_430cc7b72b1543d446e4ac9fb9a38353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_430cc7b72b1543d446e4ac9fb9a38353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_430cc7b72b1543d446e4ac9fb9a38353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1a69587b03c5a28c6e2a244584aec034(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6069, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e040a49c3f6026221bff3dd625badab6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a69587b03c5a28c6e2a244584aec034
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6069, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_20e632234844d32eadc1e9cde211f53e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 4, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3549, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9d67f52214cdd7d938a0440d03452f4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e632234844d32eadc1e9cde211f53e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3549, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_de45fc04bc245198a8b6d639ab982961(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3024, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_10bb2d54e43c0bc2da7b27135ae243ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de45fc04bc245198a8b6d639ab982961
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3024, 4, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a145880bbd22aff10323e75e568e226e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cb5bc7575b10cab7a5681e75c2ab059
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8d636a6444460b350ea8617c95e6fedd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 576, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 576, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1735c5d9d3b468c20e41f3e60f628b95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d636a6444460b350ea8617c95e6fedd
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 576, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_05a882ce170a29652e88a1d357dc5613(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1024, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_62c649dda432053ec261a95936d00a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05a882ce170a29652e88a1d357dc5613
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1024, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b29dd4d2c6675fcb5db56aa09ae22d36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b29dd4d2c6675fcb5db56aa09ae22d36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b29dd4d2c6675fcb5db56aa09ae22d36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b29dd4d2c6675fcb5db56aa09ae22d36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b29dd4d2c6675fcb5db56aa09ae22d36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b29dd4d2c6675fcb5db56aa09ae22d36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b29dd4d2c6675fcb5db56aa09ae22d36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_32261de2d6eb92f1b61d195845c9fba2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            paddle.static.InputSpec(shape=[36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00bea53e8efc70eac7cf7867a7f58281(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32261de2d6eb92f1b61d195845c9fba2
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00bea53e8efc70eac7cf7867a7f58281(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32261de2d6eb92f1b61d195845c9fba2
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a145880bbd22aff10323e75e568e226e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cb5bc7575b10cab7a5681e75c2ab059
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b1cca7dd5e8f9870b2b0813e933adf5f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[247, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[247, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_51a6283480cc9cf729d3e3151a9f766d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1cca7dd5e8f9870b2b0813e933adf5f
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5238e75ba93725039d66ac6c08fbc0cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[247, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b1dd950c54a788c7790660e6ce0c9dd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5238e75ba93725039d66ac6c08fbc0cc
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
            paddle.uniform([247, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1dd950c54a788c7790660e6ce0c9dd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5238e75ba93725039d66ac6c08fbc0cc
    def get_inputs(self):
        return [
            paddle.to_tensor(1.0, dtype='float32').reshape([]),
            paddle.uniform([247, 81], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()