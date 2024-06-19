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


class TestPrimitiveOp_f090bda775765dc18724f5316c298324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2530a4e4625af688f4e3de9927836b64
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.17802387475967407, 0.21659570932388306]], [[-0.3463458716869354, -0.28406569361686707]], [[0.057414233684539795, -0.12132707238197327]], [[0.07427680492401123, 0.1247895359992981]], [[-0.4900873303413391, 0.3299350142478943]], [[0.24544966220855713, -0.14929908514022827]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[-0.2840849459171295, 0.34122806787490845]], [[-0.10931763052940369, -0.03670382499694824]], [[0.3368977904319763, -0.48648738861083984]], [[-0.10934224724769592, 0.09921759366989136]], [[-0.09279295802116394, 0.3406224846839905]], [[-0.04301777482032776, 0.47093939781188965]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_3358d0c6c03f673eade111f4e21ea476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2530a4e4625af688f4e3de9927836b64
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.1892220377922058, 0.10756063461303711]], [[0.35799288749694824, 0.2152063250541687]], [[-0.4332648813724518, 0.4940372109413147]], [[-0.12855878472328186, -0.3311140537261963]], [[0.062365710735321045, -0.3581732511520386]], [[0.028728604316711426, 0.267437219619751]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[-0.2840849459171295, 0.34122806787490845]], [[-0.10931763052940369, -0.03670382499694824]], [[0.3368977904319763, -0.48648738861083984]], [[-0.10934224724769592, 0.09921759366989136]], [[-0.09279295802116394, 0.3406224846839905]], [[-0.04301777482032776, 0.47093939781188965]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class TestPrimitiveOp_e245c3a29892cc42a862ed60f6fb039e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38e60ad0c3265f689e713b6b70f444bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 21824, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[-0.1495116949081421, -0.23655268549919128]], [[-0.1332460343837738, 0.33323872089385986]], [[-0.014808475971221924, 0.49287915229797363]], [[-0.48712894320487976, -0.3008689880371094]], [[0.420071542263031, 0.2625719904899597]], [[0.3761158585548401, -0.15037932991981506]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class TestPrimitiveOp_a720144c415652aec4953e8b8aa304ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1546d5060eb5e7f91c606a3154f7d7bd
    def get_inputs(self):
        return [
            paddle.uniform([100, 1, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[-1.1872408390045166, -64.96479797363281, -0.7256464958190918, 1.4400469064712524], [-0.3490685820579529, -1.1309313774108887, 35.89175033569336, 1.6645225286483765]], dtype='float32').reshape([2, 4]),
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


class PrimitiveOp_85d4902e469e91bed6c2f6f0958e1106(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1542, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1542, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a8f53111e109847536093225c565595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85d4902e469e91bed6c2f6f0958e1106
    def get_inputs(self):
        return [
            paddle.uniform([1542, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1542, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_35720a4eb2ceb9ed1f0cd6cb44cbd928(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1542, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1542, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fcbdcf12b7a8666c0523b647b842aae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35720a4eb2ceb9ed1f0cd6cb44cbd928
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fcbdcf12b7a8666c0523b647b842aae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35720a4eb2ceb9ed1f0cd6cb44cbd928
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fcbdcf12b7a8666c0523b647b842aae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35720a4eb2ceb9ed1f0cd6cb44cbd928
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fcbdcf12b7a8666c0523b647b842aae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35720a4eb2ceb9ed1f0cd6cb44cbd928
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fcbdcf12b7a8666c0523b647b842aae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35720a4eb2ceb9ed1f0cd6cb44cbd928
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fcbdcf12b7a8666c0523b647b842aae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35720a4eb2ceb9ed1f0cd6cb44cbd928
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fcbdcf12b7a8666c0523b647b842aae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35720a4eb2ceb9ed1f0cd6cb44cbd928
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fcbdcf12b7a8666c0523b647b842aae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35720a4eb2ceb9ed1f0cd6cb44cbd928
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fcbdcf12b7a8666c0523b647b842aae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35720a4eb2ceb9ed1f0cd6cb44cbd928
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fcbdcf12b7a8666c0523b647b842aae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35720a4eb2ceb9ed1f0cd6cb44cbd928
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fcbdcf12b7a8666c0523b647b842aae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35720a4eb2ceb9ed1f0cd6cb44cbd928
    def get_inputs(self):
        return [
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1542, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_4a8f53111e109847536093225c565595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85d4902e469e91bed6c2f6f0958e1106
    def get_inputs(self):
        return [
            paddle.uniform([1542, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1542, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1867c065c2ff4c334fc84971cc649c84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2361, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2361, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_634ca48dc1fcb5afa39d50a448f26d95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1867c065c2ff4c334fc84971cc649c84
    def get_inputs(self):
        return [
            paddle.uniform([2361, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2361, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cc0a8e3da938349b392f7daa3bb8e5c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2361, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2361, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c067c36c73825dface430e1ea26b7997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc0a8e3da938349b392f7daa3bb8e5c9
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c067c36c73825dface430e1ea26b7997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc0a8e3da938349b392f7daa3bb8e5c9
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c067c36c73825dface430e1ea26b7997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc0a8e3da938349b392f7daa3bb8e5c9
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c067c36c73825dface430e1ea26b7997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc0a8e3da938349b392f7daa3bb8e5c9
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c067c36c73825dface430e1ea26b7997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc0a8e3da938349b392f7daa3bb8e5c9
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c067c36c73825dface430e1ea26b7997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc0a8e3da938349b392f7daa3bb8e5c9
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c067c36c73825dface430e1ea26b7997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc0a8e3da938349b392f7daa3bb8e5c9
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c067c36c73825dface430e1ea26b7997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc0a8e3da938349b392f7daa3bb8e5c9
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c067c36c73825dface430e1ea26b7997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc0a8e3da938349b392f7daa3bb8e5c9
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c067c36c73825dface430e1ea26b7997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc0a8e3da938349b392f7daa3bb8e5c9
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c067c36c73825dface430e1ea26b7997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc0a8e3da938349b392f7daa3bb8e5c9
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2361, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_634ca48dc1fcb5afa39d50a448f26d95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1867c065c2ff4c334fc84971cc649c84
    def get_inputs(self):
        return [
            paddle.uniform([2361, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2361, 4], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_af314db2abc5b50033ddc4246b9d2f7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.32846152782440186], [-0.22129765152931213], [-0.1057390570640564], [-0.43403640389442444]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4851228594779968], [-0.14451241493225098], [0.282498300075531], [0.00022339820861816406]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_38ae06577148ff3998690e6871e25968(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3450826406478882], [-0.3794686198234558], [-0.25792452692985535], [-0.4801425337791443]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.03437381982803345], [0.13219231367111206], [-0.20345637202262878], [-0.10867077112197876]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_c289a4faa1664052ea786db5e811ea4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.45028048753738403], [-0.22129765152931213], [-0.1057390570640564], [-0.43403640389442444]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.029027044773101807], [-0.14451241493225098], [0.282498300075531], [-0.09930366277694702]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_2e907f7c84906b1c93ac9e3316578505(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3894405961036682], [-0.27829545736312866], [0.195406973361969], [0.47131139039993286]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.06026780605316162], [-0.3177316188812256], [-0.24746686220169067], [-0.10867077112197876]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_0834c267b00d42c655a3fa46d6e9b445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.32846152782440186], [-0.13607582449913025], [-0.02927514910697937], [0.16101807355880737]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4851228594779968], [-0.2152409553527832], [0.26293325424194336], [0.00022339820861816406]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_f6e6a67bcd010b1cab400afa73213388(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3450826406478882], [-0.3794686198234558], [-0.25792452692985535], [-0.4801425337791443]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.03437381982803345], [0.13219231367111206], [-0.20345637202262878], [-0.20608854293823242]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_53712de0bebdbea7e9976713fb6fb477(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.11927862465381622], [-0.0435338169336319], [-0.15602411329746246], [-0.2382054328918457]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_1556a46853c7ab7df3474d60612b8b37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.45028048753738403], [-0.13607582449913025], [-0.02927514910697937], [0.16101807355880737]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.029027044773101807], [-0.2152409553527832], [0.26293325424194336], [-0.09930366277694702]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_4a926303680a59b45ad6b2c5939c2a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3894405961036682], [-0.27829545736312866], [0.195406973361969], [0.47131139039993286]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.06026780605316162], [-0.3177316188812256], [-0.24746686220169067], [-0.20608854293823242]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_376447b5fc328820a2968ec6689586ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.18944121897220612], [0.003121968824416399], [-0.1294114589691162], [0.1763419210910797]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.11927862465381622], [-0.0435338169336319], [-0.15602411329746246], [-0.2382054328918457]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_cf87b70949434646a1f8cab2522c8954(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [-0.0], [-0.0], [-0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[1.629633903503418], [14.944347381591797], [-0.20564372837543488], [2.350815773010254]], dtype='float32').reshape([4, 1]),
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


class TestPrimitiveOp_9a2f32ead46970e893f5e7dd86610a0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dfcedf316eacbbadf7dd6cacaded896
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2207202911376953]], [[-0.26910868287086487]], [[0.49499768018722534]], [[0.31563127040863037]], [[0.18310201168060303]], [[-0.06753423810005188]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.43762630224227905]], [[0.49244895577430725]], [[0.5334929823875427]], [[0.3051779568195343]], [[0.38292473554611206]], [[0.539634644985199]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_3b1e9faa37863f372ce40a9b428aa019(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dfcedf316eacbbadf7dd6cacaded896
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.11345365643501282]], [[0.42488008737564087]], [[-0.3930617570877075]], [[-0.006196498870849609]], [[-0.049382805824279785]], [[-0.4824540913105011]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.3211212158203125]], [[0.3275565803050995]], [[0.38896021246910095]], [[0.6094598770141602]], [[0.32830724120140076]], [[0.36291810870170593]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_4f90e28a399e39ec9761fd998cf713fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.32541972398757935, 0.39935123920440674, 0.44849663972854614, 0.11540669202804565, -0.284700870513916, 0.4726531505584717], dtype='float32').reshape([6]),
            paddle.to_tensor([0.06431865692138672, 0.2684096693992615, 0.29037344455718994, -0.3376345932483673, 0.031954169273376465, -0.10851988196372986], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ab3b3eaa5306f78cc0eb4186f0fbba92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.14445561170578003, -0.4791395366191864, 0.3018534183502197, 0.39743155241012573, -0.36470523476600647, -0.35002028942108154], dtype='float32').reshape([6]),
            paddle.to_tensor([0.02541947364807129, 0.2636595368385315, 0.3311959505081177, 0.2361714243888855, -0.43945392966270447, -0.1232295036315918], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_5e353d7d4fc1f2d28104361609a26580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.48757103085517883, -0.2080082893371582, -0.42475807666778564, 0.46605026721954346, 0.20001959800720215, -0.3895566165447235], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.17642048001289368, -0.2487233579158783, -0.270376980304718, -0.13760244846343994, 0.2186264991760254, -0.2595521807670593], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_01e91b993a9900cebf7095bc3192e7ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.26385489106178284, 0.4451194405555725, 0.12403160333633423, -0.2686293423175812, -0.1215813159942627, 0.21148324012756348], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.4874858558177948, -0.4435403347015381, 0.09530919790267944, -0.18070703744888306, 0.15153855085372925, 0.3255499601364136], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_46b9b0ac5afb70c079f3fe69aa069541(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.48757103085517883, -0.2080082893371582, -0.42475807666778564, 0.11540669202804565, 0.031954169273376465, -0.3895566165447235], dtype='float32').reshape([6]),
            paddle.to_tensor([0.06431865692138672, 0.2684096693992615, 0.29037344455718994, -0.13760244846343994, 0.2186264991760254, -0.10851988196372986], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_90cad7347a26ea55799e84b7a662b947(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.26385489106178284, 0.2636595368385315, 0.12403160333633423, -0.2686293423175812, -0.36470523476600647, -0.1232295036315918], dtype='float32').reshape([6]),
            paddle.to_tensor([0.02541947364807129, 0.2636595368385315, 0.3311959505081177, 0.2361714243888855, 0.15153855085372925, 0.3255499601364136], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_9f7218b4147a3de18dbeccc78fe39ed0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06431865692138672, 0.39935123920440674, 0.44849663972854614, 0.11540669202804565, 0.031954169273376465, 0.4726531505584717], dtype='float32').reshape([6]),
            paddle.to_tensor([0.06431865692138672, 0.2684096693992615, 0.29037344455718994, -0.3376345932483673, 0.031954169273376465, -0.10851988196372986], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_79459a8253ae3652cb19343e9f2db263(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.02541947364807129, 0.2636595368385315, 0.3311959505081177, 0.39743155241012573, -0.36470523476600647, -0.1232295036315918], dtype='float32').reshape([6]),
            paddle.to_tensor([0.02541947364807129, 0.2636595368385315, 0.3311959505081177, 0.2361714243888855, -0.43945392966270447, -0.1232295036315918], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_b4fc105b28e2c1e13d5085e795f0935a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.06958289444446564, 0.036181844770908356, -0.004434196278452873, 0.0199829563498497, 0.0050819143652915955, 0.014829179272055626], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, -0.0, 0.0, -0.0, 0.0, 0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ecf30eefecc6b48e9d422da37bab1bf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.1305505335330963, 0.3338804543018341, 0.36943504214286804, -0.11111395061016083, -0.12637335062026978, 0.1820666342973709], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.33199575543403625, -0.22836582362651825, -0.34756752848625183, 0.16422390937805176, 0.20932304859161377, -0.3245543837547302], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_76bf28d576471f78e903c8d03a8ad80d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.05951806902885437, -0.10773999989032745, 0.3165246844291687, 0.3168014883995056, -0.40207958221435547, -0.23662489652633667], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.3756703734397888, 0.0007895529270172119, 0.10967040061950684, -0.22466818988323212, 0.014978617429733276, 0.2685166001319885], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_d83ee5c277a84c1f42819df80e09791e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06431865692138672, 0.39935123920440674, 0.44849663972854614, 0.46605026721954346, 0.20001959800720215, 0.4726531505584717], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.17642048001289368, -0.2487233579158783, -0.270376980304718, -0.3376345932483673, 0.031954169273376465, -0.2595521807670593], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_1a3d439f457661e66c6dd1f71a766410(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.02541947364807129, 0.4451194405555725, 0.3311959505081177, 0.39743155241012573, -0.1215813159942627, 0.21148324012756348], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.4874858558177948, -0.4435403347015381, 0.09530919790267944, -0.18070703744888306, -0.43945392966270447, -0.1232295036315918], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_37df4509fe053100ce098d0364a9e6ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.9476150274276733, 0.04578424245119095, -1.3868508338928223, -1.4261629581451416, 0.06802213191986084, 0.8506050705909729], dtype='float32').reshape([6]),
            paddle.to_tensor([1.1597552299499512, -0.1744886040687561, -1.3873158693313599, 1.2288305759429932, -1.3389828205108643, -1.1987411975860596], dtype='float32').reshape([6]),
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


class TestPrimitiveOp_3678382c3a715d9568b60d2bfe8524b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29eb514882c92ab3a86517239fc1edf9
    def get_inputs(self):
        return [
            paddle.uniform([300, 1, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[-1.0668749809265137, 1.4476252794265747, 6.671823501586914, -0.573180079460144], [-3.0731329917907715, 0.8413622975349426, -1.51128351688385, 0.7123610973358154]], dtype='float32').reshape([2, 4]),
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


class TestPrimitiveOp_f49298a118280dd468939e2691330bf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4603fe0ce7ef18f56a1834167dd2c8c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 16384], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[0.4999712109565735], [0.499972939491272], [0.4999868869781494], [0.49994540214538574], [0.49996715784072876], [0.49995970726013184], [0.4996270537376404], [0.49993979930877686], [0.49999815225601196], [0.49994999170303345], [0.49994152784347534], [0.4999924302101135], [0.4999188184738159], [0.4999881982803345], [0.49996745586395264], [0.49998897314071655], [0.4998822808265686], [0.4999808669090271], [0.4999426603317261], [0.4998893141746521], [0.49984049797058105]]], dtype='float32').reshape([1, 21, 1]),
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


class TestPrimitiveOp_aba79e762b5bddd8ee9fda26fd1b09f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.233851820230484], [-0.4165400564670563], [-0.028367280960083008], [-0.4339827001094818], [-0.40483978390693665], [-0.08868327736854553]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.443808376789093], [0.451566219329834], [0.11306852102279663], [0.21094638109207153], [0.34457260370254517], [0.4592204689979553]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_257827aa098415e2a5a7e551487fd6d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.47951558232307434], [0.2852780222892761], [-0.3642190098762512], [-0.16062521934509277], [-0.4562195837497711], [-0.22439396381378174]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.28020238876342773], [0.33490562438964844], [0.39741456508636475], [0.40055209398269653], [0.46220719814300537], [0.0712430477142334]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_b972eee929199602931386e74ff4de01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4716532230377197], [-0.4165400564670563], [-0.0014056563377380371], [0.09141331911087036], [-0.2677716016769409], [-0.08868327736854553]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.443808376789093], [0.451566219329834], [-0.028237223625183105], [-0.2575671076774597], [0.07678425312042236], [0.023420095443725586]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_92bbe55bf3816551e8c6b7a4ce858960(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16195690631866455], [0.2852780222892761], [-0.3642190098762512], [-0.16062521934509277], [-0.4562195837497711], [-0.22439396381378174]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.28020238876342773], [-0.30021917819976807], [0.39741456508636475], [-0.4255662262439728], [0.46220719814300537], [-0.33687111735343933]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_4291778fe51f679dedf60cba54540112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.233851820230484], [0.1492787003517151], [-0.028367280960083008], [-0.4339827001094818], [-0.40483978390693665], [0.01156306266784668]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.16058611869812012], [-0.26714545488357544], [0.11306852102279663], [0.21094638109207153], [0.34457260370254517], [0.4592204689979553]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_6dff4bbe39121cdae73247329900dd1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.47951558232307434], [0.4709428548812866], [0.08498668670654297], [0.024811983108520508], [0.25165116786956787], [0.1815364956855774]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.4044957756996155], [0.33490562438964844], [0.13769322633743286], [0.40055209398269653], [0.0017930269241333008], [0.0712430477142334]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_01320b2a257e1f42aa816aa219d393d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.041902512311935425], [-0.4516246020793915], [-0.012981231324374676], [0.33478492498397827], [0.12920251488685608], [-0.06198274716734886]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_159d708a2f4b133facccd9a7ea982ddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4716532230377197], [0.1492787003517151], [-0.0014056563377380371], [0.09141331911087036], [-0.2677716016769409], [0.01156306266784668]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.16058611869812012], [-0.26714545488357544], [-0.028237223625183105], [-0.2575671076774597], [0.07678425312042236], [0.023420095443725586]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_b44a24d95e50c66cfe11432cd32206aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16195690631866455], [0.4709428548812866], [0.08498668670654297], [0.024811983108520508], [0.25165116786956787], [0.1815364956855774]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.4044957756996155], [-0.30021917819976807], [0.13769322633743286], [-0.4255662262439728], [0.0017930269241333008], [-0.33687111735343933]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_10748bf1e98511da4cd1f855ac3e3747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17620480060577393], [0.3211304843425751], [-0.0014141988940536976], [0.15717318654060364], [-0.086090087890625], [-0.006146775558590889]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.041902512311935425], [-0.4516246020793915], [-0.012981231324374676], [0.33478492498397827], [0.12920251488685608], [-0.06198274716734886]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_3d95ee17d2af71ae24a3c71263fe446d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [-0.0], [-0.0], [0.0], [0.0], [-0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.7621942758560181], [2.406358480453491], [-8.179211616516113], [-1.1300383806228638], [2.5007827281951904], [-9.083782196044922]], dtype='float32').reshape([6, 1]),
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


class TestPrimitiveOp_d46604715552b1b931193656776ed162(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_473768ad5406d82abfbdd94c8502a0d7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
            paddle.to_tensor([0.30443012714385986, 0.18485862016677856, 0.27507275342941284, -0.2047252357006073], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_a22415b4e71d15395831bf0ad3816b86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_473768ad5406d82abfbdd94c8502a0d7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.30443012714385986, 0.18485862016677856, 0.27507275342941284, -0.2047252357006073], dtype='float32').reshape([4]),
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


class PrimitiveOp_00b4936f08fe886c1c94210a59657112(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2053, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2053, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0fa0dd014b71e50a487a93e61a4cb277(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00b4936f08fe886c1c94210a59657112
    def get_inputs(self):
        return [
            paddle.uniform([2053, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2053, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_79f0024fcf273ac4ae2b936f524f0261(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2053, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2053, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_79cfe44a4e98cc13abb671558be0d75f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79f0024fcf273ac4ae2b936f524f0261
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_79cfe44a4e98cc13abb671558be0d75f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79f0024fcf273ac4ae2b936f524f0261
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_79cfe44a4e98cc13abb671558be0d75f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79f0024fcf273ac4ae2b936f524f0261
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_79cfe44a4e98cc13abb671558be0d75f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79f0024fcf273ac4ae2b936f524f0261
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_79cfe44a4e98cc13abb671558be0d75f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79f0024fcf273ac4ae2b936f524f0261
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_79cfe44a4e98cc13abb671558be0d75f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79f0024fcf273ac4ae2b936f524f0261
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_79cfe44a4e98cc13abb671558be0d75f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79f0024fcf273ac4ae2b936f524f0261
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_79cfe44a4e98cc13abb671558be0d75f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79f0024fcf273ac4ae2b936f524f0261
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_79cfe44a4e98cc13abb671558be0d75f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79f0024fcf273ac4ae2b936f524f0261
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_79cfe44a4e98cc13abb671558be0d75f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79f0024fcf273ac4ae2b936f524f0261
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_79cfe44a4e98cc13abb671558be0d75f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79f0024fcf273ac4ae2b936f524f0261
    def get_inputs(self):
        return [
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2053, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_0fa0dd014b71e50a487a93e61a4cb277(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00b4936f08fe886c1c94210a59657112
    def get_inputs(self):
        return [
            paddle.uniform([2053, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2053, 4], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_b6103b9f32cf3b2b1ffb9b5428976f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_042c31aed200f7f6f007e22c52c71138
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
            paddle.to_tensor([0.4355631470680237, 0.465285062789917, -0.430935263633728, -0.3135901987552643, 0.17000901699066162, 0.14725714921951294, -0.48706769943237305, -0.4442668557167053, 0.15751606225967407, -0.1798284351825714, -0.3336707353591919, 0.42167603969573975, 0.17849797010421753, 0.0706479549407959, -0.47450143098831177, -0.12924033403396606, -0.4698745906352997, 0.48139137029647827, 0.45962780714035034, 0.15411460399627686], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_515ac4efae6db22fe7a2f424e29f9fb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_042c31aed200f7f6f007e22c52c71138
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4355631470680237, 0.465285062789917, -0.430935263633728, -0.3135901987552643, 0.17000901699066162, 0.14725714921951294, -0.48706769943237305, -0.4442668557167053, 0.15751606225967407, -0.1798284351825714, -0.3336707353591919, 0.42167603969573975, 0.17849797010421753, 0.0706479549407959, -0.47450143098831177, -0.12924033403396606, -0.4698745906352997, 0.48139137029647827, 0.45962780714035034, 0.15411460399627686], dtype='float32').reshape([20]),
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


class TestPrimitiveOp_05adaa582a3ee79322cfe91a26c0e993(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.1422344446182251], [0.027947425842285156], [-0.41162997484207153], [-0.38247421383857727], [-0.3881467580795288]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.2685706615447998], [-0.292533814907074], [0.31706804037094116], [0.401347815990448], [0.06959110498428345]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_b2195ccb0e6c0faec8afea37317271ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.4251890182495117], [0.2107974886894226], [-0.14475178718566895], [-0.1629328727722168], [-0.3283495306968689]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.327716588973999], [0.043894171714782715], [0.10123854875564575], [0.042978525161743164], [0.14828330278396606]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_71428dc7f829988ea0f4ec99d4253a9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09874492883682251], [0.027947425842285156], [-0.41162997484207153], [-0.38247421383857727], [0.24568265676498413]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.2685706615447998], [-0.42384102940559387], [-0.11871618032455444], [-0.3788006603717804], [0.06959110498428345]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_0a56f811a04307b965da6b64e4d0a02e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.05882182717323303], [0.31854957342147827], [-0.14475178718566895], [-0.1629328727722168], [0.2213781476020813]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.185139000415802], [0.043894171714782715], [-0.4893096685409546], [-0.23725301027297974], [0.14828330278396606]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_be22d041a07d96e4edd3fa58ebce94d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.1422344446182251], [0.3373464345932007], [0.1920177936553955], [0.2963539958000183], [-0.3881467580795288]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.2158190906047821], [-0.292533814907074], [0.31706804037094116], [0.401347815990448], [0.0006369352340698242]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_8771ec2691cc820a356ecffe8fe34fae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.4251890182495117], [0.2107974886894226], [0.30673420429229736], [0.45866304636001587], [-0.3283495306968689]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.327716588973999], [-0.12176045775413513], [0.10123854875564575], [0.042978525161743164], [-0.41153308749198914]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_944a1475ac530572ed42b161fc6a839e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.013971466571092606], [0.3335578441619873], [-0.1266230344772339], [-0.043917324393987656], [-0.01946902647614479]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0], [0.053489383310079575], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_505aae32ebd92399148bda45490234f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09874492883682251], [0.3373464345932007], [0.1920177936553955], [0.2963539958000183], [0.24568265676498413]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.2158190906047821], [-0.42384102940559387], [-0.11871618032455444], [-0.3788006603717804], [0.0006369352340698242]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_4d0e8910c0904a0d6f74d09c4b002443(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.05882182717323303], [0.31854957342147827], [0.30673420429229736], [0.45866304636001587], [0.2213781476020813]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.185139000415802], [-0.12176045775413513], [-0.4893096685409546], [-0.23725301027297974], [-0.41153308749198914]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_5307ce93306579d84d3daa59547e0a7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0767413005232811], [0.33515846729278564], [0.24735787510871887], [0.46985098719596863], [0.15509217977523804]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.013971466571092606], [0.28006845712661743], [-0.1266230344772339], [-0.043917324393987656], [-0.01946902647614479]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_65f03bba0ba630eb4b811eea87138de9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [0.19098681211471558], [-0.0], [-0.0], [-0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.8179406523704529], [0.16437004506587982], [1.5119022130966187], [1.093470811843872], [1.1255319118499756]], dtype='float32').reshape([5, 1]),
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


class PrimitiveOp_d8f0c5fb70f121203f16f9b8a76b27ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1825, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1825, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8039dfd134ec8251f4526f00de306d2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8f0c5fb70f121203f16f9b8a76b27ed
    def get_inputs(self):
        return [
            paddle.uniform([1825, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1825, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e5c336dc1aabe147e74fdb1d5a3e4008(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1825, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1825, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eacfcf1bcb0ba73b637e6b817ea0e34f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5c336dc1aabe147e74fdb1d5a3e4008
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eacfcf1bcb0ba73b637e6b817ea0e34f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5c336dc1aabe147e74fdb1d5a3e4008
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eacfcf1bcb0ba73b637e6b817ea0e34f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5c336dc1aabe147e74fdb1d5a3e4008
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eacfcf1bcb0ba73b637e6b817ea0e34f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5c336dc1aabe147e74fdb1d5a3e4008
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eacfcf1bcb0ba73b637e6b817ea0e34f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5c336dc1aabe147e74fdb1d5a3e4008
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eacfcf1bcb0ba73b637e6b817ea0e34f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5c336dc1aabe147e74fdb1d5a3e4008
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eacfcf1bcb0ba73b637e6b817ea0e34f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5c336dc1aabe147e74fdb1d5a3e4008
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eacfcf1bcb0ba73b637e6b817ea0e34f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5c336dc1aabe147e74fdb1d5a3e4008
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eacfcf1bcb0ba73b637e6b817ea0e34f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5c336dc1aabe147e74fdb1d5a3e4008
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eacfcf1bcb0ba73b637e6b817ea0e34f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5c336dc1aabe147e74fdb1d5a3e4008
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eacfcf1bcb0ba73b637e6b817ea0e34f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5c336dc1aabe147e74fdb1d5a3e4008
    def get_inputs(self):
        return [
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1825, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_8039dfd134ec8251f4526f00de306d2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8f0c5fb70f121203f16f9b8a76b27ed
    def get_inputs(self):
        return [
            paddle.uniform([1825, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1825, 4], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_a716b85d24a2cfcf4ad3dfc2d0a15391(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0368fad43623b73e790d949bddeb3e4a
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
            paddle.to_tensor([-0.48465049266815186, -0.04762253165245056, 0.23276299238204956, -0.06612768769264221, -0.49176597595214844, -0.06729069352149963, 0.3730962872505188, -0.010074377059936523, 0.39248335361480713, -0.21158581972122192, 0.41349464654922485, 0.32767152786254883, 0.4794207811355591, 0.046976566314697266, 0.16938143968582153, 0.22565364837646484], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_5bbba335662f4fd4afd5275400a1edc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0368fad43623b73e790d949bddeb3e4a
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.48465049266815186, -0.04762253165245056, 0.23276299238204956, -0.06612768769264221, -0.49176597595214844, -0.06729069352149963, 0.3730962872505188, -0.010074377059936523, 0.39248335361480713, -0.21158581972122192, 0.41349464654922485, 0.32767152786254883, 0.4794207811355591, 0.046976566314697266, 0.16938143968582153, 0.22565364837646484], dtype='float32').reshape([16]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([16]),
        ]


class PrimitiveOp_8246830632bd0f1920591b0c75a39378(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4832], dtype='float32'),
            paddle.static.InputSpec(shape=[4832], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_62df3c133fc36ce00d327df4bdc5e5cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8246830632bd0f1920591b0c75a39378
    def get_inputs(self):
        return [
            paddle.uniform([4832], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4832], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_62df3c133fc36ce00d327df4bdc5e5cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8246830632bd0f1920591b0c75a39378
    def get_inputs(self):
        return [
            paddle.uniform([4832], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4832], dtype='float32', min=-0.5, max=0.5),
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


class PrimitiveOp_4662b2cb65e93df1a3e3988c9496ef51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3087, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[3087, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9656924445f93cb8736d55ef4cf979a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4662b2cb65e93df1a3e3988c9496ef51
    def get_inputs(self):
        return [
            paddle.uniform([3087, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3087, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_16f563eeab026b1438c6cc0a8483a2ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3087, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3087, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_973a90a4533d7fc546d2cc4402333c06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16f563eeab026b1438c6cc0a8483a2ee
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_973a90a4533d7fc546d2cc4402333c06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16f563eeab026b1438c6cc0a8483a2ee
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_973a90a4533d7fc546d2cc4402333c06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16f563eeab026b1438c6cc0a8483a2ee
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_973a90a4533d7fc546d2cc4402333c06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16f563eeab026b1438c6cc0a8483a2ee
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_973a90a4533d7fc546d2cc4402333c06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16f563eeab026b1438c6cc0a8483a2ee
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_973a90a4533d7fc546d2cc4402333c06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16f563eeab026b1438c6cc0a8483a2ee
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_973a90a4533d7fc546d2cc4402333c06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16f563eeab026b1438c6cc0a8483a2ee
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_973a90a4533d7fc546d2cc4402333c06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16f563eeab026b1438c6cc0a8483a2ee
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_973a90a4533d7fc546d2cc4402333c06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16f563eeab026b1438c6cc0a8483a2ee
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_973a90a4533d7fc546d2cc4402333c06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16f563eeab026b1438c6cc0a8483a2ee
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_973a90a4533d7fc546d2cc4402333c06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16f563eeab026b1438c6cc0a8483a2ee
    def get_inputs(self):
        return [
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3087, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_f9656924445f93cb8736d55ef4cf979a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4662b2cb65e93df1a3e3988c9496ef51
    def get_inputs(self):
        return [
            paddle.uniform([3087, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3087, 4], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_db1c15669a960d610e1b58978fe55126(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2119961678981781], [-0.3788200914859772], [-0.4864160418510437], [-0.1263948678970337], [0.3646824359893799], [-0.39871159195899963], [-0.35914939641952515], [-0.22272902727127075], [0.22541850805282593]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.0015003085136413574], [0.4079332947731018], [0.06591081619262695], [0.09166866540908813], [-0.16269496083259583], [0.3435378670692444], [0.3831961750984192], [0.1379859447479248], [-0.07388219237327576]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_22d1f7166838a47c35236e0977fda177(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.3213467597961426], [-0.23528355360031128], [0.43103843927383423], [-0.39259082078933716], [-0.278251588344574], [-0.40452665090560913], [-0.044628649950027466], [-0.009750127792358398], [-0.073464035987854]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.14492911100387573], [0.09219878911972046], [0.11862069368362427], [-0.2647143602371216], [0.35444319248199463], [0.4993610978126526], [0.15277761220932007], [0.09750252962112427], [0.4061077833175659]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_9ba26fe7f8b0f71a478b0d588fa29eee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2119961678981781], [-0.3788200914859772], [-0.04915744066238403], [0.328036367893219], [0.3646824359893799], [-0.39871159195899963], [-0.35914939641952515], [0.16614854335784912], [0.22541850805282593]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.1698777675628662], [0.3688810467720032], [0.06591081619262695], [0.09166866540908813], [-0.16269496083259583], [-0.0751321017742157], [-0.13586968183517456], [0.1379859447479248], [-0.4187920093536377]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_e5d861530883335c850b5dff4ef978a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.3213467597961426], [-0.23528355360031128], [0.44690072536468506], [0.41063904762268066], [-0.278251588344574], [-0.40452665090560913], [-0.044628649950027466], [0.17216962575912476], [0.09485393762588501]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.08901798725128174], [0.0880575180053711], [0.11862069368362427], [-0.3751475214958191], [0.35444319248199463], [0.4993610978126526], [-0.4423360228538513], [0.09750252962112427], [-0.12597641348838806]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_cee0a856ba355c0f4ab3eb316190e638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11566966772079468], [0.32729101181030273], [-0.4864160418510437], [-0.1263948678970337], [0.4190939664840698], [0.11164069175720215], [0.20763516426086426], [-0.22272902727127075], [0.49746018648147583]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.0015003085136413574], [0.4079332947731018], [-0.20244035124778748], [-0.11076605319976807], [-0.1853780746459961], [0.3435378670692444], [0.3831961750984192], [-0.0418873131275177], [-0.07388219237327576]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_b6dcaf86782542f0b69ecadce3b0d114(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3514004349708557], [0.19362658262252808], [0.43103843927383423], [-0.39259082078933716], [0.17742151021957397], [-0.18762415647506714], [-0.008309662342071533], [-0.009750127792358398], [-0.073464035987854]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.14492911100387573], [0.09219878911972046], [0.06880456209182739], [-0.2647143602371216], [0.103046715259552], [-0.24344509840011597], [0.15277761220932007], [-0.4944602847099304], [0.4061077833175659]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_6995712314d1cd76cf366f2441cc1bf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.04147614538669586], [0.23358313739299774], [-0.1406402289867401], [0.18773312866687775], [-0.28871142864227295], [0.27953481674194336], [-0.060519345104694366], [-0.08555299043655396], [-0.13173845410346985]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_b4cdff333a2b0b9f5ddc2669bd18985a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11566966772079468], [0.32729101181030273], [-0.04915744066238403], [0.328036367893219], [0.4190939664840698], [0.11164069175720215], [0.20763516426086426], [0.16614854335784912], [0.49746018648147583]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.1698777675628662], [0.3688810467720032], [-0.20244035124778748], [-0.11076605319976807], [-0.1853780746459961], [-0.0751321017742157], [-0.13586968183517456], [-0.0418873131275177], [-0.4187920093536377]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_acba8c9b20b6aa7e9b651083988ddefc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3514004349708557], [0.19362658262252808], [0.44690072536468506], [0.41063904762268066], [0.17742151021957397], [-0.18762415647506714], [-0.008309662342071533], [0.17216962575912476], [0.09485393762588501]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.08901798725128174], [0.0880575180053711], [0.06880456209182739], [-0.3751475214958191], [0.103046715259552], [-0.24344509840011597], [-0.4423360228538513], [-0.4944602847099304], [-0.12597641348838806]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_9d61ee3d3cc913749ac5a8a7a07f6bf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07492263615131378], [-0.004390621092170477], [0.057955678552389145], [0.3448050618171692], [0.044957485049963], [0.01042583305388689], [0.14909015595912933], [0.13868293166160583], [0.20233629643917084]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.04147614538669586], [0.23358313739299774], [-0.1406402289867401], [0.18773312866687775], [-0.28871142864227295], [0.27953481674194336], [-0.060519345104694366], [-0.08555299043655396], [-0.13173845410346985]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_20319fd0b350121a7cace1f782c387a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [-0.0], [0.0], [-0.0], [0.0], [-0.0], [-0.0], [-0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.44641369581222534], [54.200477600097656], [3.4266860485076904], [0.4555383622646332], [7.421876907348633], [-25.811748504638672], [1.4059245586395264], [1.6168962717056274], [1.6510865688323975]], dtype='float32').reshape([9, 1]),
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


class TestPrimitiveOp_1dd119ea5143f700016ddea0d09612f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_429e2045e6f4da7e01c19b5d8020bfd2
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 32768], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[0.4999780058860779], [0.4999953508377075], [0.49997788667678833], [0.4999668002128601], [0.4999018907546997], [0.4999576210975647], [0.49995797872543335], [0.499999463558197], [0.4999748468399048], [0.4999490976333618], [0.4999680519104004], [0.49997979402542114], [0.49997955560684204], [0.499960720539093], [0.49999791383743286], [0.49997490644454956], [0.4999961256980896], [0.4999915361404419], [0.4999672770500183]]], dtype='float32').reshape([1, 19, 1]),
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


class TestPrimitiveOp_df14878d32bc94d9bc3603c7ff7c281a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dfcedf316eacbbadf7dd6cacaded896
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.18202835321426392]], [[-0.30439454317092896]], [[-0.004671335220336914]], [[-0.3094761371612549]], [[-0.02697235345840454]], [[0.1652316451072693]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.8063583970069885]], [[0.48542889952659607]], [[0.3626091778278351]], [[0.5082688927650452]], [[0.7265392541885376]], [[0.4818327724933624]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_f9d7baad8e51c1ac35b65c4192e0c951(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dfcedf316eacbbadf7dd6cacaded896
    def get_inputs(self):
        return [
            paddle.to_tensor([[[-0.05138421058654785]], [[-0.12731075286865234]], [[0.35545605421066284]], [[0.42558956146240234]], [[-0.4715643525123596]], [[-0.3462280035018921]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.5605470538139343]], [[0.5314770936965942]], [[0.7138208150863647]], [[0.32816851139068604]], [[0.32900112867355347]], [[0.3850231468677521]]], dtype='float32').reshape([6, 1, 1]),
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


class PrimitiveOp_ecca134bd7e1a60f250fdd54729ea9fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2119, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2119, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2dc053ec50bf26ed46079705a37da748(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecca134bd7e1a60f250fdd54729ea9fd
    def get_inputs(self):
        return [
            paddle.uniform([2119, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2119, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_64e11f3788e71620189b45d4a0be493e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2119, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2119, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_83e850f0a04becb1383b9b1803f433a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64e11f3788e71620189b45d4a0be493e
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83e850f0a04becb1383b9b1803f433a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64e11f3788e71620189b45d4a0be493e
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83e850f0a04becb1383b9b1803f433a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64e11f3788e71620189b45d4a0be493e
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83e850f0a04becb1383b9b1803f433a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64e11f3788e71620189b45d4a0be493e
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83e850f0a04becb1383b9b1803f433a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64e11f3788e71620189b45d4a0be493e
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83e850f0a04becb1383b9b1803f433a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64e11f3788e71620189b45d4a0be493e
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83e850f0a04becb1383b9b1803f433a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64e11f3788e71620189b45d4a0be493e
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83e850f0a04becb1383b9b1803f433a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64e11f3788e71620189b45d4a0be493e
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83e850f0a04becb1383b9b1803f433a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64e11f3788e71620189b45d4a0be493e
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83e850f0a04becb1383b9b1803f433a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64e11f3788e71620189b45d4a0be493e
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83e850f0a04becb1383b9b1803f433a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64e11f3788e71620189b45d4a0be493e
    def get_inputs(self):
        return [
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2119, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_2dc053ec50bf26ed46079705a37da748(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecca134bd7e1a60f250fdd54729ea9fd
    def get_inputs(self):
        return [
            paddle.uniform([2119, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2119, 4], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_f1cbc8a634d738efaedf57ac12a3114b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.050851911306381226]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.21787577867507935]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_4bd70231a50d9dd95ca4665db11ae919(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.204064279794693]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.1975727379322052]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_f1cbc8a634d738efaedf57ac12a3114b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.050851911306381226]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.21787577867507935]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_71204ce6a6db5cda73e17b1314689c8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3629520535469055]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.4216543436050415]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_04ac1ee6882d6bb2df40f589c0c5b8d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0921517014503479]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.41818875074386597]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_4bd70231a50d9dd95ca4665db11ae919(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.204064279794693]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.1975727379322052]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_036c7812b435a67a8e3f46c22a2404b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12773509323596954]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_04ac1ee6882d6bb2df40f589c0c5b8d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0921517014503479]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.41818875074386597]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_71204ce6a6db5cda73e17b1314689c8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3629520535469055]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.4216543436050415]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_1c80bca53564a1516c645df592a2bd90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.40041637420654297]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.12773509323596954]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_6e0fc7dbdb795141d3b821a29850e8c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.6809943914413452]], dtype='float32').reshape([1, 1]),
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


class PrimitiveOp_425cb9fddeac84e399e211596403b05b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5606, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[5606, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d1668f9aa54d9a112e956711ca86589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_425cb9fddeac84e399e211596403b05b
    def get_inputs(self):
        return [
            paddle.uniform([5606, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5606, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c8c11a3e8ef289081d7e5a262427364e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5606, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5606, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b1adcb677f4638e8506747b1ffbd503d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8c11a3e8ef289081d7e5a262427364e
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1adcb677f4638e8506747b1ffbd503d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8c11a3e8ef289081d7e5a262427364e
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1adcb677f4638e8506747b1ffbd503d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8c11a3e8ef289081d7e5a262427364e
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1adcb677f4638e8506747b1ffbd503d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8c11a3e8ef289081d7e5a262427364e
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1adcb677f4638e8506747b1ffbd503d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8c11a3e8ef289081d7e5a262427364e
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1adcb677f4638e8506747b1ffbd503d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8c11a3e8ef289081d7e5a262427364e
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1adcb677f4638e8506747b1ffbd503d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8c11a3e8ef289081d7e5a262427364e
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1adcb677f4638e8506747b1ffbd503d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8c11a3e8ef289081d7e5a262427364e
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1adcb677f4638e8506747b1ffbd503d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8c11a3e8ef289081d7e5a262427364e
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1adcb677f4638e8506747b1ffbd503d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8c11a3e8ef289081d7e5a262427364e
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1adcb677f4638e8506747b1ffbd503d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8c11a3e8ef289081d7e5a262427364e
    def get_inputs(self):
        return [
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5606, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_0d1668f9aa54d9a112e956711ca86589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_425cb9fddeac84e399e211596403b05b
    def get_inputs(self):
        return [
            paddle.uniform([5606, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5606, 4], dtype='float32', min=-0.5, max=0.5),
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


class PrimitiveOp_cb2105c8e5c5a86960a10ace80bcebc2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1036, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1036, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c33f4c60df4c9f3275aa3a7d7b74620a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb2105c8e5c5a86960a10ace80bcebc2
    def get_inputs(self):
        return [
            paddle.uniform([1036, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1036, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b87631c3f6acdb61d83c557406dfaf30(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1036, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1036, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b7864012665fa009a85c10b61be858c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b87631c3f6acdb61d83c557406dfaf30
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b7864012665fa009a85c10b61be858c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b87631c3f6acdb61d83c557406dfaf30
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b7864012665fa009a85c10b61be858c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b87631c3f6acdb61d83c557406dfaf30
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b7864012665fa009a85c10b61be858c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b87631c3f6acdb61d83c557406dfaf30
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b7864012665fa009a85c10b61be858c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b87631c3f6acdb61d83c557406dfaf30
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b7864012665fa009a85c10b61be858c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b87631c3f6acdb61d83c557406dfaf30
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b7864012665fa009a85c10b61be858c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b87631c3f6acdb61d83c557406dfaf30
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b7864012665fa009a85c10b61be858c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b87631c3f6acdb61d83c557406dfaf30
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b7864012665fa009a85c10b61be858c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b87631c3f6acdb61d83c557406dfaf30
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b7864012665fa009a85c10b61be858c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b87631c3f6acdb61d83c557406dfaf30
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b7864012665fa009a85c10b61be858c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b87631c3f6acdb61d83c557406dfaf30
    def get_inputs(self):
        return [
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1036, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_c33f4c60df4c9f3275aa3a7d7b74620a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb2105c8e5c5a86960a10ace80bcebc2
    def get_inputs(self):
        return [
            paddle.uniform([1036, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1036, 4], dtype='float32', min=-0.5, max=0.5),
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


class PrimitiveOp_ff1d0f3b349bc7b1f5a7e3eaed12756d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1809, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1809, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2fccb19288ce5535791a86746a499916(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff1d0f3b349bc7b1f5a7e3eaed12756d
    def get_inputs(self):
        return [
            paddle.uniform([1809, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1809, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e66bb9eb292cad9c160c7429865dad5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1809, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1809, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_65dc660ebed38bf42f56711ebf1c5255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e66bb9eb292cad9c160c7429865dad5d
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65dc660ebed38bf42f56711ebf1c5255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e66bb9eb292cad9c160c7429865dad5d
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65dc660ebed38bf42f56711ebf1c5255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e66bb9eb292cad9c160c7429865dad5d
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65dc660ebed38bf42f56711ebf1c5255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e66bb9eb292cad9c160c7429865dad5d
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65dc660ebed38bf42f56711ebf1c5255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e66bb9eb292cad9c160c7429865dad5d
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65dc660ebed38bf42f56711ebf1c5255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e66bb9eb292cad9c160c7429865dad5d
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65dc660ebed38bf42f56711ebf1c5255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e66bb9eb292cad9c160c7429865dad5d
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65dc660ebed38bf42f56711ebf1c5255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e66bb9eb292cad9c160c7429865dad5d
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65dc660ebed38bf42f56711ebf1c5255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e66bb9eb292cad9c160c7429865dad5d
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65dc660ebed38bf42f56711ebf1c5255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e66bb9eb292cad9c160c7429865dad5d
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65dc660ebed38bf42f56711ebf1c5255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e66bb9eb292cad9c160c7429865dad5d
    def get_inputs(self):
        return [
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1809, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_2fccb19288ce5535791a86746a499916(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff1d0f3b349bc7b1f5a7e3eaed12756d
    def get_inputs(self):
        return [
            paddle.uniform([1809, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1809, 4], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_95dda8c142e16696f964c58d3782e6cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0df867a4644c834ec9fd270790bdf2c
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
            paddle.to_tensor([0.03167968988418579, -0.37636035680770874, -0.21670317649841309, 0.39032089710235596, -0.44819122552871704, 0.4351775646209717, 0.23851841688156128, -0.3821527063846588, -0.28143811225891113, 0.29411834478378296, 0.40033769607543945, -0.4059436023235321, 0.4736446142196655, -0.3660675883293152, 0.04382580518722534, -0.24653342366218567, 0.4353167414665222, 0.2468177080154419, -0.38413405418395996, -0.42135077714920044, 0.36219167709350586, 0.0969361662864685, 0.03515017032623291, 0.04207485914230347], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_eb8117575b9282b6b32376451272dd32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0df867a4644c834ec9fd270790bdf2c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03167968988418579, -0.37636035680770874, -0.21670317649841309, 0.39032089710235596, -0.44819122552871704, 0.4351775646209717, 0.23851841688156128, -0.3821527063846588, -0.28143811225891113, 0.29411834478378296, 0.40033769607543945, -0.4059436023235321, 0.4736446142196655, -0.3660675883293152, 0.04382580518722534, -0.24653342366218567, 0.4353167414665222, 0.2468177080154419, -0.38413405418395996, -0.42135077714920044, 0.36219167709350586, 0.0969361662864685, 0.03515017032623291, 0.04207485914230347], dtype='float32').reshape([24]),
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


class PrimitiveOp_89c46dde32a063ad716a7dcab8b0d6e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17421], dtype='float32'),
            paddle.static.InputSpec(shape=[17421], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_87e0e5c2e74644e3569fb8fb9e7f3316(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89c46dde32a063ad716a7dcab8b0d6e6
    def get_inputs(self):
        return [
            paddle.uniform([17421], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([17421], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_87e0e5c2e74644e3569fb8fb9e7f3316(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89c46dde32a063ad716a7dcab8b0d6e6
    def get_inputs(self):
        return [
            paddle.uniform([17421], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([17421], dtype='float32', min=-0.5, max=0.5),
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


class PrimitiveOp_80ab61bab4d51821f580b604c23d8b22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4179, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[4179, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b5e5cc0b6428ed5f4b9983b5fd0ddc00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ab61bab4d51821f580b604c23d8b22
    def get_inputs(self):
        return [
            paddle.uniform([4179, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4179, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a2ceccd03abacc4784874515608887f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4179, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4179, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_312beda43b0e6dafac62445785db8b4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2ceccd03abacc4784874515608887f9
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_312beda43b0e6dafac62445785db8b4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2ceccd03abacc4784874515608887f9
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_312beda43b0e6dafac62445785db8b4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2ceccd03abacc4784874515608887f9
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_312beda43b0e6dafac62445785db8b4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2ceccd03abacc4784874515608887f9
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_312beda43b0e6dafac62445785db8b4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2ceccd03abacc4784874515608887f9
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_312beda43b0e6dafac62445785db8b4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2ceccd03abacc4784874515608887f9
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_312beda43b0e6dafac62445785db8b4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2ceccd03abacc4784874515608887f9
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_312beda43b0e6dafac62445785db8b4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2ceccd03abacc4784874515608887f9
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_312beda43b0e6dafac62445785db8b4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2ceccd03abacc4784874515608887f9
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_312beda43b0e6dafac62445785db8b4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2ceccd03abacc4784874515608887f9
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_312beda43b0e6dafac62445785db8b4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2ceccd03abacc4784874515608887f9
    def get_inputs(self):
        return [
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4179, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_b5e5cc0b6428ed5f4b9983b5fd0ddc00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ab61bab4d51821f580b604c23d8b22
    def get_inputs(self):
        return [
            paddle.uniform([4179, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4179, 4], dtype='float32', min=-0.5, max=0.5),
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


class PrimitiveOp_884c7d65ecff1ea3fefc3d423d18de5e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4662, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[4662, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dbb86f8061b23545c67cabb593b95ee7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884c7d65ecff1ea3fefc3d423d18de5e
    def get_inputs(self):
        return [
            paddle.uniform([4662, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4662, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fd0ccad6a1a2f721559733339ebb99a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4662, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4662, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef7c1b27906b8ec664fc68e4bf2b1864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd0ccad6a1a2f721559733339ebb99a6
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef7c1b27906b8ec664fc68e4bf2b1864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd0ccad6a1a2f721559733339ebb99a6
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef7c1b27906b8ec664fc68e4bf2b1864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd0ccad6a1a2f721559733339ebb99a6
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef7c1b27906b8ec664fc68e4bf2b1864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd0ccad6a1a2f721559733339ebb99a6
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef7c1b27906b8ec664fc68e4bf2b1864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd0ccad6a1a2f721559733339ebb99a6
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef7c1b27906b8ec664fc68e4bf2b1864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd0ccad6a1a2f721559733339ebb99a6
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef7c1b27906b8ec664fc68e4bf2b1864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd0ccad6a1a2f721559733339ebb99a6
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef7c1b27906b8ec664fc68e4bf2b1864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd0ccad6a1a2f721559733339ebb99a6
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef7c1b27906b8ec664fc68e4bf2b1864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd0ccad6a1a2f721559733339ebb99a6
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef7c1b27906b8ec664fc68e4bf2b1864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd0ccad6a1a2f721559733339ebb99a6
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef7c1b27906b8ec664fc68e4bf2b1864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd0ccad6a1a2f721559733339ebb99a6
    def get_inputs(self):
        return [
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4662, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_dbb86f8061b23545c67cabb593b95ee7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_884c7d65ecff1ea3fefc3d423d18de5e
    def get_inputs(self):
        return [
            paddle.uniform([4662, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4662, 4], dtype='float32', min=-0.5, max=0.5),
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


class PrimitiveOp_c592440547f6931a996a25469b303970(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3857, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[3857, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57477ab0661891352adde0985894bbd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c592440547f6931a996a25469b303970
    def get_inputs(self):
        return [
            paddle.uniform([3857, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3857, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4955bb2476210f83eb132909ecfd94ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3857, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3857, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_50717b5fa89452c7447a09c5aa15a421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4955bb2476210f83eb132909ecfd94ba
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_50717b5fa89452c7447a09c5aa15a421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4955bb2476210f83eb132909ecfd94ba
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_50717b5fa89452c7447a09c5aa15a421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4955bb2476210f83eb132909ecfd94ba
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_50717b5fa89452c7447a09c5aa15a421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4955bb2476210f83eb132909ecfd94ba
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_50717b5fa89452c7447a09c5aa15a421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4955bb2476210f83eb132909ecfd94ba
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_50717b5fa89452c7447a09c5aa15a421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4955bb2476210f83eb132909ecfd94ba
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_50717b5fa89452c7447a09c5aa15a421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4955bb2476210f83eb132909ecfd94ba
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_50717b5fa89452c7447a09c5aa15a421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4955bb2476210f83eb132909ecfd94ba
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_50717b5fa89452c7447a09c5aa15a421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4955bb2476210f83eb132909ecfd94ba
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_50717b5fa89452c7447a09c5aa15a421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4955bb2476210f83eb132909ecfd94ba
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_50717b5fa89452c7447a09c5aa15a421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4955bb2476210f83eb132909ecfd94ba
    def get_inputs(self):
        return [
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3857, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_57477ab0661891352adde0985894bbd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c592440547f6931a996a25469b303970
    def get_inputs(self):
        return [
            paddle.uniform([3857, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3857, 4], dtype='float32', min=-0.5, max=0.5),
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