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



class PrimitiveOp_0ab26114d7983355cd961d75de1b302b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9dd7e31bde95dc8eaf0c003735d4449(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.12378641217947006]], [[0.03100680187344551]], [[0.3605198264122009]], [[0.25344493985176086]], [[0.1395755559206009]], [[0.3745681345462799]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.7648835182189941]], [[0.5925349593162537]], [[0.7611966729164124]], [[0.5286526679992676]], [[0.7556431293487549]], [[0.6973389983177185]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_24e77ec1515f8ced39e46e0ac8b0c1a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.26165881752967834]], [[0.11104203015565872]], [[0.37972795963287354]], [[0.4138566553592682]], [[0.2808537483215332]], [[0.42193496227264404]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.605302095413208]], [[0.5831876993179321]], [[0.7477464079856873]], [[0.6850302219390869]], [[0.6371867060661316]], [[0.5999490022659302]]], dtype='float32').reshape([6, 1, 1]),
        ]


class PrimitiveOp_adcea79a7d32530c808b3fbffdecda65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b45340e00eb3a4b1af26b6965795dfbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b45340e00eb3a4b1af26b6965795dfbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b45340e00eb3a4b1af26b6965795dfbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b45340e00eb3a4b1af26b6965795dfbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b45340e00eb3a4b1af26b6965795dfbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b45340e00eb3a4b1af26b6965795dfbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b45340e00eb3a4b1af26b6965795dfbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f2e561418f620b48e4bd798deb678dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75b3ec42bd74d23da03ca936ee8b7f7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75b3ec42bd74d23da03ca936ee8b7f7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75b3ec42bd74d23da03ca936ee8b7f7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75b3ec42bd74d23da03ca936ee8b7f7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75b3ec42bd74d23da03ca936ee8b7f7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75b3ec42bd74d23da03ca936ee8b7f7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75b3ec42bd74d23da03ca936ee8b7f7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f2ba88ba8184ce9bc6a4140eb6f9897(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2359d1766c4658acb24afe71ea57a121(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12096, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf3fa8836de3729a95ac0da64729abaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2359d1766c4658acb24afe71ea57a121
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4a112d08f8eecb192e097cac8b4a275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4a112d08f8eecb192e097cac8b4a275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4a112d08f8eecb192e097cac8b4a275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4a112d08f8eecb192e097cac8b4a275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4a112d08f8eecb192e097cac8b4a275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4a112d08f8eecb192e097cac8b4a275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4a112d08f8eecb192e097cac8b4a275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6452b5d2f4d21a3545083c26bf5c0fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6452b5d2f4d21a3545083c26bf5c0fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6452b5d2f4d21a3545083c26bf5c0fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6452b5d2f4d21a3545083c26bf5c0fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6452b5d2f4d21a3545083c26bf5c0fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6452b5d2f4d21a3545083c26bf5c0fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6452b5d2f4d21a3545083c26bf5c0fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e67498f16e12748959cd5c1913fd4ba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cb9d719725306a63a050fe5cf6b020b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.45568397641181946, 0.3022495210170746]], [[0.17868947982788086, 0.3039349615573883]], [[0.4016329348087311, 0.06874442845582962]], [[0.28851118683815, 0.2187298983335495]], [[0.3890455663204193, 0.2713296115398407]], [[0.4850843846797943, 0.13000766932964325]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.2777520418167114, 0.1937178373336792]], [[0.21845121681690216, 0.16587240993976593]], [[0.4173831343650818, 0.4494837820529938]], [[0.33341124653816223, 0.2507529556751251]], [[0.022635046392679214, 0.15895231068134308]], [[0.23398567736148834, 0.18906551599502563]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_eed30c2e9345e305e75cf145b993efbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.05417050048708916, 0.37919148802757263]], [[0.0902252122759819, 0.4723469316959381]], [[0.3709464967250824, 0.05640269070863724]], [[0.16880078613758087, 0.32737284898757935]], [[0.44946685433387756, 0.18289893865585327]], [[0.2071576565504074, 0.2513531446456909]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.2777520418167114, 0.1937178373336792]], [[0.21845121681690216, 0.16587240993976593]], [[0.4173831343650818, 0.4494837820529938]], [[0.33341124653816223, 0.2507529556751251]], [[0.022635046392679214, 0.15895231068134308]], [[0.23398567736148834, 0.18906551599502563]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class PrimitiveOp_8b0071a67648dd3072ffe70503722fe9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5ba5e47f8a6411cb005987e7e35efa5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b0071a67648dd3072ffe70503722fe9
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.39800122380256653, 0.28543126583099365]], [[0.2436889410018921, 0.4663369059562683]], [[0.19892513751983643, 0.29967978596687317]], [[0.4644017219543457, 0.19519776105880737]], [[0.007041141390800476, 0.19483619928359985]], [[0.42418187856674194, 0.14996834099292755]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_30bb6fd5af91ce7657d492a7e1cae8c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30bb6fd5af91ce7657d492a7e1cae8c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30bb6fd5af91ce7657d492a7e1cae8c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30bb6fd5af91ce7657d492a7e1cae8c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30bb6fd5af91ce7657d492a7e1cae8c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30bb6fd5af91ce7657d492a7e1cae8c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30bb6fd5af91ce7657d492a7e1cae8c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_436b8b7edaa5cc5ed65ce2d9d7ec6ec8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
            paddle.to_tensor([0.17627683281898499, 0.010097586549818516, 0.2177966833114624, 0.36453622579574585, 0.34911927580833435, 0.009820925071835518, 0.432222843170166, 0.15748214721679688, 0.16535866260528564, 0.18357667326927185, 0.2256554216146469, 0.16378404200077057, 0.46310481429100037, 0.25582805275917053, 0.23355577886104584, 0.3437178134918213], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_8480994efa06d336de4f0e82e52d99f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.17627683281898499, 0.010097586549818516, 0.2177966833114624, 0.36453622579574585, 0.34911927580833435, 0.009820925071835518, 0.432222843170166, 0.15748214721679688, 0.16535866260528564, 0.18357667326927185, 0.2256554216146469, 0.16378404200077057, 0.46310481429100037, 0.25582805275917053, 0.23355577886104584, 0.3437178134918213], dtype='float32').reshape([16]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_7364b3bae43a394c6eef01fc601610e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7364b3bae43a394c6eef01fc601610e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7364b3bae43a394c6eef01fc601610e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7364b3bae43a394c6eef01fc601610e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7364b3bae43a394c6eef01fc601610e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7364b3bae43a394c6eef01fc601610e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7364b3bae43a394c6eef01fc601610e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_cfaf53e08f16ced2b7c0f7e98fad27c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b820b60b96edeb6b2c4753edd23b40b
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cfaf53e08f16ced2b7c0f7e98fad27c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b820b60b96edeb6b2c4753edd23b40b
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b45340e00eb3a4b1af26b6965795dfbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b45340e00eb3a4b1af26b6965795dfbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b45340e00eb3a4b1af26b6965795dfbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b45340e00eb3a4b1af26b6965795dfbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b45340e00eb3a4b1af26b6965795dfbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b45340e00eb3a4b1af26b6965795dfbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b45340e00eb3a4b1af26b6965795dfbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78fb6ccbd33d2bd15825b9b21e3aecc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78fb6ccbd33d2bd15825b9b21e3aecc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78fb6ccbd33d2bd15825b9b21e3aecc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78fb6ccbd33d2bd15825b9b21e3aecc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78fb6ccbd33d2bd15825b9b21e3aecc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78fb6ccbd33d2bd15825b9b21e3aecc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78fb6ccbd33d2bd15825b9b21e3aecc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_966a7b1aeae197afcd65fc7da74524a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99d04deefd3715f422cc68ba7061f3c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99d04deefd3715f422cc68ba7061f3c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99d04deefd3715f422cc68ba7061f3c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99d04deefd3715f422cc68ba7061f3c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99d04deefd3715f422cc68ba7061f3c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99d04deefd3715f422cc68ba7061f3c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99d04deefd3715f422cc68ba7061f3c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_0a9d45312078191191bef560ef308a66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_13f41ebfd8962ac69f094f95a6a6a457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13f41ebfd8962ac69f094f95a6a6a457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13f41ebfd8962ac69f094f95a6a6a457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13f41ebfd8962ac69f094f95a6a6a457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13f41ebfd8962ac69f094f95a6a6a457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13f41ebfd8962ac69f094f95a6a6a457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13f41ebfd8962ac69f094f95a6a6a457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13f41ebfd8962ac69f094f95a6a6a457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13f41ebfd8962ac69f094f95a6a6a457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13f41ebfd8962ac69f094f95a6a6a457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13f41ebfd8962ac69f094f95a6a6a457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_26374218f165833153f20543fd3cfd0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2931b855774f639216adbc1d5caa2743(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26374218f165833153f20543fd3cfd0b
    def get_inputs(self):
        return [
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e550e076a243ef110a84edc2744e886e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_01132c0096e8de61f14b0f7ec33cb8bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e550e076a243ef110a84edc2744e886e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a9d45312078191191bef560ef308a66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1ba9192fa475d78ed36c9b1af8f4f84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.494537889957428, 0.32202088832855225, 0.052795205265283585, 0.36225536465644836], [0.17443254590034485, 0.1808221936225891, 0.2965104281902313, 0.09832663089036942], [0.4867327809333801, 0.07026968151330948, 0.3219892382621765, 0.04937509074807167], [0.08756934106349945, 0.04703647270798683, 0.17231324315071106, 0.2765970230102539], [0.20844760537147522, 0.3499521017074585, 0.1441584676504135, 0.4825325012207031]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.0026577040553092957, 0.035248465836048126, 0.11444555968046188, 0.07061809301376343], [0.007260100916028023, 0.4510287642478943, 0.07052305340766907, 0.1170760914683342], [0.09106604009866714, 0.03159948065876961, 0.2624412178993225, 0.40884068608283997], [0.4145248830318451, 0.25235515832901, 0.38111579418182373, 0.04465979337692261], [0.10431944578886032, 0.056864410638809204, 0.4997289180755615, 0.30007919669151306]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_146a165acf7c715aa0314a1892c59985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_146a165acf7c715aa0314a1892c59985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_146a165acf7c715aa0314a1892c59985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_146a165acf7c715aa0314a1892c59985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_146a165acf7c715aa0314a1892c59985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_146a165acf7c715aa0314a1892c59985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_146a165acf7c715aa0314a1892c59985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32a6a06ed6b412dcdc7c86c555445b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32a6a06ed6b412dcdc7c86c555445b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32a6a06ed6b412dcdc7c86c555445b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32a6a06ed6b412dcdc7c86c555445b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32a6a06ed6b412dcdc7c86c555445b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32a6a06ed6b412dcdc7c86c555445b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32a6a06ed6b412dcdc7c86c555445b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3e277da4799f04d89245e851c0774e82(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 5376, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4267b5b58b2f0d3f7f9c351095d21b1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e277da4799f04d89245e851c0774e82
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_708b49733002c3fbd307b23c818846a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.005050296895205975, 0.022361747920513153, 0.46100813150405884, 0.06284592300653458], [0.010459013283252716, 0.09524784982204437, 0.026963822543621063, 0.039656467735767365], [0.35230201482772827, 0.4778911769390106, 0.25436726212501526, 0.3508223593235016], [0.010459013283252716, 0.09524784982204437, 0.026963822543621063, 0.039656467735767365], [0.35230201482772827, 0.4778911769390106, 0.25436726212501526, 0.3508223593235016]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.29338374733924866, 0.3130703270435333, 0.2226981520652771, 0.15290579199790955], [0.004089932888746262, 0.3951409161090851, 0.4945432245731354, 0.46871089935302734], [0.004426018334925175, 0.16711683571338654, 0.4394293427467346, 0.165910542011261], [0.004089932888746262, 0.3951409161090851, 0.4945432245731354, 0.46871089935302734], [0.004426018334925175, 0.16711683571338654, 0.4394293427467346, 0.165910542011261]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_36ab29e79a434a6f89b8806af898711a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36ab29e79a434a6f89b8806af898711a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36ab29e79a434a6f89b8806af898711a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36ab29e79a434a6f89b8806af898711a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36ab29e79a434a6f89b8806af898711a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36ab29e79a434a6f89b8806af898711a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36ab29e79a434a6f89b8806af898711a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17722cd741deeb0cdbf8fe85b557172c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17722cd741deeb0cdbf8fe85b557172c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17722cd741deeb0cdbf8fe85b557172c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17722cd741deeb0cdbf8fe85b557172c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17722cd741deeb0cdbf8fe85b557172c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17722cd741deeb0cdbf8fe85b557172c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17722cd741deeb0cdbf8fe85b557172c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_435a9d3c1ce303cd1a8a3ec2845a3c61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.24578014016151428], [0.06570521742105484], [0.08684854209423065], [0.12536022067070007], [0.05390194058418274], [0.2627392113208771], [0.33099016547203064], [0.15110880136489868], [0.10229241102933884]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.331709086894989], [0.4315752387046814], [0.41960468888282776], [0.3730376958847046], [0.39346981048583984], [0.15195290744304657], [0.4746951162815094], [0.42640355229377747], [0.43999600410461426]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_219687bb7adf0504d94acabcb27a0c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0795329138636589], [0.10759332776069641], [0.14820031821727753], [0.020885052159428596], [0.17430728673934937], [0.19093403220176697], [0.2295200675725937], [0.07001930475234985], [0.03557616472244263]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.37394869327545166], [0.4309453070163727], [0.26047682762145996], [0.45039767026901245], [0.4338553249835968], [0.27303066849708557], [0.35420629382133484], [0.35356906056404114], [0.4564360976219177]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_d78341378c816e8fc1770500f9a55ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3287186026573181], [0.18020182847976685], [0.08684854209423065], [0.2041427493095398], [0.05390194058418274], [0.2627392113208771], [0.33099016547203064], [0.15110880136489868], [0.10229241102933884]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.331709086894989], [0.4315752387046814], [0.41960468888282776], [0.3054076135158539], [0.3308812975883484], [0.10524086654186249], [0.22144579887390137], [0.42640355229377747], [0.1348181813955307]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_611e885f6c7e114eb90f530ad6a6d00b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33190086483955383], [0.10759332776069641], [0.42710933089256287], [0.49739205837249756], [0.17430728673934937], [0.4577294886112213], [0.2397078424692154], [0.07001930475234985], [0.03557616472244263]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.20715078711509705], [0.11182703822851181], [0.1563730388879776], [0.45039767026901245], [0.01656464673578739], [0.03139396384358406], [0.338609904050827], [0.35356906056404114], [0.4564360976219177]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_25468afaa34900ff1c36ee3e5516263a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.24578014016151428], [0.06570521742105484], [0.45486047863960266], [0.12536022067070007], [0.46158698201179504], [0.43161243200302124], [0.4087907373905182], [0.27460622787475586], [0.4056161344051361]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.04048364609479904], [0.24893270432949066], [0.0918775126338005], [0.3730376958847046], [0.39346981048583984], [0.15195290744304657], [0.4746951162815094], [0.14278598129749298], [0.43999600410461426]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_a1054d95cbd600c8cc71fee536437892(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0795329138636589], [0.2186332643032074], [0.14820031821727753], [0.020885052159428596], [0.27499523758888245], [0.19093403220176697], [0.2295200675725937], [0.4841283857822418], [0.15295632183551788]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.37394869327545166], [0.4309453070163727], [0.26047682762145996], [0.11842523515224457], [0.4338553249835968], [0.27303066849708557], [0.35420629382133484], [0.2772969901561737], [0.2206316888332367]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_1cc992e26be6b927ec04779c33cdf7f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.060815583914518356], [0.039965640753507614], [-0.1308436244726181], [0.019399628043174744], [-0.05451255291700363], [0.04418803006410599], [-0.0026167957112193108], [0.10532432794570923], [0.016015464439988136]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_877488033590d110c5ad90ae24f50c83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3287186026573181], [0.18020182847976685], [0.45486047863960266], [0.2041427493095398], [0.46158698201179504], [0.43161243200302124], [0.4087907373905182], [0.27460622787475586], [0.4056161344051361]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.04048364609479904], [0.24893270432949066], [0.0918775126338005], [0.3054076135158539], [0.3308812975883484], [0.10524086654186249], [0.22144579887390137], [0.14278598129749298], [0.1348181813955307]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_c35929b7da8aa87d8b43fcd30509660e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33190086483955383], [0.2186332643032074], [0.42710933089256287], [0.49739205837249756], [0.27499523758888245], [0.4577294886112213], [0.2397078424692154], [0.4841283857822418], [0.15295632183551788]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.20715078711509705], [0.11182703822851181], [0.1563730388879776], [0.11842523515224457], [0.01656464673578739], [0.03139396384358406], [0.338609904050827], [0.2772969901561737], [0.2206316888332367]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_958c81d02563c6afd2fc19d11a9ee02b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03595733270049095], [-0.007340885233134031], [0.0982726514339447], [-0.03837602213025093], [0.033778347074985504], [0.13914377987384796], [-0.018528800457715988], [0.027264565229415894], [-0.018326351419091225]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.060815583914518356], [0.039965640753507614], [-0.1308436244726181], [0.019399628043174744], [-0.05451255291700363], [0.04418803006410599], [-0.0026167957112193108], [0.10532432794570923], [0.016015464439988136]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_2d1afe391a92205debf46061ca659034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [0.0], [-0.0], [0.0], [-0.0], [0.0], [-0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[2.691326379776001], [6.444253444671631], [2.331434726715088], [1.5055142641067505], [2.6138312816619873], [0.6824290156364441], [0.8587713837623596], [-2.8630480766296387], [1.8739036321640015]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_b683a7538b123f277ffef55f48129604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dedf6206a0081c6b26f57c3fe8c334e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3079172372817993]], [[0.49177154898643494]], [[0.20487146079540253]], [[0.1812247484922409]], [[0.4014056622982025]], [[0.45381712913513184]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.5946179032325745]], [[0.7134573459625244]], [[0.7299712300300598]], [[0.6914341449737549]], [[0.7282995581626892]], [[0.8240033984184265]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_44df1947b0fcc19d28b1e9f658d37b54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2525706887245178]], [[0.157998725771904]], [[0.28690123558044434]], [[0.08029963821172714]], [[0.08474474400281906]], [[0.04007527232170105]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.5491977334022522]], [[0.7433127760887146]], [[0.5438257455825806]], [[0.6047941446304321]], [[0.6819583773612976]], [[0.593007504940033]]], dtype='float32').reshape([6, 1, 1]),
        ]


class PrimitiveOp_2d17218c50c61309e237e3456e91e44b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_13727e8d36dbe65dde3d3911b1d7fee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d17218c50c61309e237e3456e91e44b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c94935bc06c4368bbd7babecc97eb90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([5504, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_590b76150af74a9040c1a1ef3be723ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_590b76150af74a9040c1a1ef3be723ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_590b76150af74a9040c1a1ef3be723ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_590b76150af74a9040c1a1ef3be723ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_590b76150af74a9040c1a1ef3be723ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_590b76150af74a9040c1a1ef3be723ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_590b76150af74a9040c1a1ef3be723ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_590b76150af74a9040c1a1ef3be723ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_590b76150af74a9040c1a1ef3be723ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_590b76150af74a9040c1a1ef3be723ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_590b76150af74a9040c1a1ef3be723ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54b40a97c0f0745c4db1d2c07941ece5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26374218f165833153f20543fd3cfd0b
    def get_inputs(self):
        return [
            paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90c3e8435317b182878ecc90c1d33380(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e550e076a243ef110a84edc2744e886e
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c94935bc06c4368bbd7babecc97eb90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([5504, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eac074c22cfd5e8f037bc292f8208161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.04639088734984398, 0.02690395526587963, 0.023448554798960686, 0.22740550339221954], [0.3495735824108124, 0.28023090958595276, 0.017169106751680374, 0.40246057510375977], [0.24151821434497833, 0.058896105736494064, 0.48122143745422363, 0.40122267603874207], [0.3495735824108124, 0.28023090958595276, 0.017169106751680374, 0.40246057510375977], [0.24151821434497833, 0.058896105736494064, 0.48122143745422363, 0.40122267603874207], [0.09526802599430084, 0.17614632844924927, 0.44104236364364624, 0.26393523812294006], [0.09526802599430084, 0.17614632844924927, 0.44104236364364624, 0.26393523812294006]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.18214254081249237, 0.4342849552631378, 0.4673704504966736, 0.1338675171136856], [0.3171866536140442, 0.060204219073057175, 0.45422884821891785, 0.4605034291744232], [0.46335697174072266, 0.48995307087898254, 0.17117619514465332, 0.27034786343574524], [0.3171866536140442, 0.060204219073057175, 0.45422884821891785, 0.4605034291744232], [0.46335697174072266, 0.48995307087898254, 0.17117619514465332, 0.27034786343574524], [0.3005690574645996, 0.4370850622653961, 0.1850690394639969, 0.4781649112701416], [0.3005690574645996, 0.4370850622653961, 0.1850690394639969, 0.4781649112701416]], dtype='float32').reshape([7, 4]),
        ]


class TestPrimitiveOp_e9027683390446ba9a09511d9a8d5a7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9027683390446ba9a09511d9a8d5a7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f88ac26fb5b292f335c7a7d9c9c4828(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28516e44e4a792f606fc0120dd0015ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_320fa3c9d4da345503b4efd696226334(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14632879197597504, 0.329461008310318, 0.20341289043426514, 0.2538435161113739, 0.2506648898124695, 0.06819789856672287], dtype='float32').reshape([6]),
            paddle.to_tensor([0.12323445081710815, 0.3513852059841156, 0.4086536467075348, 0.1541879028081894, 0.0009443100425414741, 0.38373863697052], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_21f7010dfa211b152fdae24f409c2f94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.41780155897140503, 0.042055390775203705, 0.2534020245075226, 0.06820432841777802, 0.27422139048576355, 0.3214428424835205], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4688446819782257, 0.11460484564304352, 0.44962406158447266, 0.3697131276130676, 0.20614933967590332, 0.04639579355716705], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_0a3b17541a859e599d8a59b99b17d146(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44832438230514526, 0.10885077714920044, 0.4067733883857727, 0.3167877793312073, 0.16312891244888306, 0.40700143575668335], dtype='float32').reshape([6]),
            paddle.to_tensor([0.24217796325683594, 0.3652135729789734, 0.3323742747306824, 0.19348189234733582, 0.23804683983325958, 0.18084128201007843], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ae4c1eed4884174a152a21d8e468ad43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09651322662830353, 0.20735882222652435, 0.14432112872600555, 0.38802286982536316, 0.4720301926136017, 0.22569142282009125], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3871614336967468, 0.34158778190612793, 0.2761411964893341, 0.46681585907936096, 0.3594697415828705, 0.42803919315338135], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_21214c48b02c860b849447cf37959a87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14632879197597504, 0.10885077714920044, 0.4067733883857727, 0.2538435161113739, 0.16312891244888306, 0.38373863697052], dtype='float32').reshape([6]),
            paddle.to_tensor([0.24217796325683594, 0.3652135729789734, 0.4086536467075348, 0.19348189234733582, 0.23804683983325958, 0.38373863697052], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_22da4a919a78ecfa118ea855f32f0ed6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09651322662830353, 0.11460484564304352, 0.14432112872600555, 0.3697131276130676, 0.27422139048576355, 0.22569142282009125], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4688446819782257, 0.34158778190612793, 0.44962406158447266, 0.46681585907936096, 0.3594697415828705, 0.42803919315338135], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_3a54eb05459fd241c47c52e695308b29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14632879197597504, 0.3513852059841156, 0.4086536467075348, 0.2538435161113739, 0.2506648898124695, 0.38373863697052], dtype='float32').reshape([6]),
            paddle.to_tensor([0.12323445081710815, 0.3513852059841156, 0.4086536467075348, 0.1541879028081894, 0.0009443100425414741, 0.38373863697052], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_8c72cd43fc74eed4a2688e835e2be9db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4688446819782257, 0.11460484564304352, 0.44962406158447266, 0.3697131276130676, 0.27422139048576355, 0.3214428424835205], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4688446819782257, 0.11460484564304352, 0.44962406158447266, 0.3697131276130676, 0.20614933967590332, 0.04639579355716705], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_e6d850ba30feeec906fb3fcec85ee8ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.059916090220212936, 0.03441131114959717, -0.009807296097278595, -0.009715639054775238, 0.008566196076571941, -0.04576300457119942], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, 0.0, 0.0, -0.0, 0.0, -0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_92ca956fa127c71041bc2a9a60dfb8dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1347816288471222, 0.3404231071472168, 0.30603325366973877, 0.20401570200920105, 0.125804603099823, 0.22596827149391174], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3452511727809906, 0.23703217506408691, 0.36957383155822754, 0.25513482093811035, 0.20058786869049072, 0.2939213514328003], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_c429a6035a163aae61675f91a6323658(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44332313537597656, 0.07833011448383331, 0.3515130281448364, 0.21895873546600342, 0.24018536508083344, 0.18391931056976318], dtype='float32').reshape([6]),
            paddle.to_tensor([0.24183732271194458, 0.27447330951690674, 0.21023115515708923, 0.42741936445236206, 0.4157499670982361, 0.3268653154373169], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_09b4666bed49393796d119fb7b896d29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44832438230514526, 0.3513852059841156, 0.4086536467075348, 0.3167877793312073, 0.2506648898124695, 0.40700143575668335], dtype='float32').reshape([6]),
            paddle.to_tensor([0.12323445081710815, 0.3513852059841156, 0.3323742747306824, 0.1541879028081894, 0.0009443100425414741, 0.18084128201007843], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_8fd3ce1df5f0aec1576298eb16f5579a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4688446819782257, 0.20735882222652435, 0.44962406158447266, 0.38802286982536316, 0.4720301926136017, 0.3214428424835205], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3871614336967468, 0.11460484564304352, 0.2761411964893341, 0.3697131276130676, 0.20614933967590332, 0.04639579355716705], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_7097d34f2e49dfab11f10222069af373(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.6169165968894958, 1.0884554386138916, -0.513830840587616, -1.0021898746490479, -0.5872495770454407, -0.8409115672111511], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.4248875379562378, 0.2934707999229431, 0.8078591227531433, -0.3192192018032074, 1.3046693801879883, -0.853856086730957], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ae3f3f8b6ecff02efcf49c324d7e4476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1811, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d17ff9a53cb6c5c2ba72082eccf9654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d17ff9a53cb6c5c2ba72082eccf9654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d17ff9a53cb6c5c2ba72082eccf9654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d17ff9a53cb6c5c2ba72082eccf9654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d17ff9a53cb6c5c2ba72082eccf9654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d17ff9a53cb6c5c2ba72082eccf9654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d17ff9a53cb6c5c2ba72082eccf9654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d17ff9a53cb6c5c2ba72082eccf9654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d17ff9a53cb6c5c2ba72082eccf9654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d17ff9a53cb6c5c2ba72082eccf9654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d17ff9a53cb6c5c2ba72082eccf9654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2931b855774f639216adbc1d5caa2743(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26374218f165833153f20543fd3cfd0b
    def get_inputs(self):
        return [
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01132c0096e8de61f14b0f7ec33cb8bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e550e076a243ef110a84edc2744e886e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae3f3f8b6ecff02efcf49c324d7e4476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1811, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9b96e0b62629f7b7fae9bfa17e5529be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8400, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e44f331e9df1607f84b5bb77658a4970(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b96e0b62629f7b7fae9bfa17e5529be
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9fccdeff17dd3385de7600032eed84cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1171131432056427, 0.3744145631790161, 0.2392769455909729, 0.2158881276845932, 0.25468266010284424, 0.14356206357479095, 0.33271417021751404, 0.11324551701545715, 0.4381972551345825, 0.1540030539035797, 0.2635442912578583, 0.4736747741699219, 0.13221971690654755, 0.003876861184835434, 0.442859411239624, 0.3939966559410095, 0.14856456220149994, 0.14854028820991516, 0.34614554047584534, 0.1592680811882019, 0.46174877882003784, 0.07964815944433212, 0.3786410093307495, 0.15283527970314026], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_96ece5ead04e913065ad1e1391e907a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1171131432056427, 0.3744145631790161, 0.2392769455909729, 0.2158881276845932, 0.25468266010284424, 0.14356206357479095, 0.33271417021751404, 0.11324551701545715, 0.4381972551345825, 0.1540030539035797, 0.2635442912578583, 0.4736747741699219, 0.13221971690654755, 0.003876861184835434, 0.442859411239624, 0.3939966559410095, 0.14856456220149994, 0.14854028820991516, 0.34614554047584534, 0.1592680811882019, 0.46174877882003784, 0.07964815944433212, 0.3786410093307495, 0.15283527970314026], dtype='float32').reshape([24]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_6452b5d2f4d21a3545083c26bf5c0fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6452b5d2f4d21a3545083c26bf5c0fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6452b5d2f4d21a3545083c26bf5c0fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6452b5d2f4d21a3545083c26bf5c0fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6452b5d2f4d21a3545083c26bf5c0fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6452b5d2f4d21a3545083c26bf5c0fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6452b5d2f4d21a3545083c26bf5c0fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32a6a06ed6b412dcdc7c86c555445b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32a6a06ed6b412dcdc7c86c555445b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32a6a06ed6b412dcdc7c86c555445b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32a6a06ed6b412dcdc7c86c555445b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32a6a06ed6b412dcdc7c86c555445b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32a6a06ed6b412dcdc7c86c555445b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32a6a06ed6b412dcdc7c86c555445b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c86df241ca52f0e237d322eec16b7fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1559, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_290cc801afc44f995eb6c72d28600b01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_290cc801afc44f995eb6c72d28600b01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_290cc801afc44f995eb6c72d28600b01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_290cc801afc44f995eb6c72d28600b01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_290cc801afc44f995eb6c72d28600b01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_290cc801afc44f995eb6c72d28600b01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_290cc801afc44f995eb6c72d28600b01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_290cc801afc44f995eb6c72d28600b01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_290cc801afc44f995eb6c72d28600b01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_290cc801afc44f995eb6c72d28600b01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_290cc801afc44f995eb6c72d28600b01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ae06657d4d252920239f4880d46dccf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26374218f165833153f20543fd3cfd0b
    def get_inputs(self):
        return [
            paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe8ac241d60488e16773ffdde01283c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e550e076a243ef110a84edc2744e886e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c86df241ca52f0e237d322eec16b7fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1559, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7364b3bae43a394c6eef01fc601610e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7364b3bae43a394c6eef01fc601610e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7364b3bae43a394c6eef01fc601610e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7364b3bae43a394c6eef01fc601610e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7364b3bae43a394c6eef01fc601610e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7364b3bae43a394c6eef01fc601610e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7364b3bae43a394c6eef01fc601610e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a77236391752492f8833a61bf2caff3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
            paddle.to_tensor([0.04089539870619774, 0.0974174216389656, 0.28220126032829285, 0.3516616225242615], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_44729ea3111982fc3784d9e36bf64b47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.04089539870619774, 0.0974174216389656, 0.28220126032829285, 0.3516616225242615], dtype='float32').reshape([4]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([4]),
        ]


class PrimitiveOp_c6235af744836bd7eab2c8afc32a7be0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_54b922740ca860bf392dba1ce8daca1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6235af744836bd7eab2c8afc32a7be0
    def get_inputs(self):
        return [
            paddle.to_tensor([4], dtype='int32').reshape([1]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_a0c8cc47b0a606acfc2b66d0bd5c4283(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6235af744836bd7eab2c8afc32a7be0
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_75b3ec42bd74d23da03ca936ee8b7f7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75b3ec42bd74d23da03ca936ee8b7f7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75b3ec42bd74d23da03ca936ee8b7f7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75b3ec42bd74d23da03ca936ee8b7f7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75b3ec42bd74d23da03ca936ee8b7f7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75b3ec42bd74d23da03ca936ee8b7f7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75b3ec42bd74d23da03ca936ee8b7f7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7c5b443afad064c54df8d3e98a1a00e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3244754672050476, 0.08234945684671402, 0.4903813898563385, 0.31153324246406555], [0.08088105171918869, 0.3441753089427948, 0.004226476885378361, 0.3615778982639313], [0.14430546760559082, 0.1698196679353714, 0.0353219099342823, 0.45951274037361145], [0.004541008733212948, 0.25024741888046265, 0.09216580539941788, 0.37754079699516296], [0.004541008733212948, 0.25024741888046265, 0.09216580539941788, 0.37754079699516296], [0.14430546760559082, 0.1698196679353714, 0.0353219099342823, 0.45951274037361145]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.011722530238330364, 0.23153094947338104, 0.09831975400447845, 0.030159827321767807], [0.14865818619728088, 0.02919861115515232, 0.4017900824546814, 0.12034225463867188], [0.14986677467823029, 0.18789073824882507, 0.19580401480197906, 0.2937736213207245], [0.04571889713406563, 0.22461551427841187, 0.2299029678106308, 0.25975126028060913], [0.04571889713406563, 0.22461551427841187, 0.2299029678106308, 0.25975126028060913], [0.14986677467823029, 0.18789073824882507, 0.19580401480197906, 0.2937736213207245]], dtype='float32').reshape([6, 4]),
        ]


class TestPrimitiveOp_25a0c583261987bf017c84f08ecea5fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16867926716804504, 0.4793153405189514, 0.06113574281334877, 0.3498322665691376], [0.2094501107931137, 0.10077470541000366, 0.4583267569541931, 0.35940390825271606], [0.28303632140159607, 0.36188018321990967, 0.2449793815612793, 0.43129047751426697], [0.34789857268333435, 0.0024067636113613844, 0.1131611317396164, 0.11788292974233627], [0.16867926716804504, 0.4793153405189514, 0.06113574281334877, 0.3498322665691376]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.3900073766708374, 0.2156796008348465, 0.14067278802394867, 0.45257124304771423], [0.3895103335380554, 0.40887120366096497, 0.4178506135940552, 0.32785341143608093], [0.16654981672763824, 0.3116723299026489, 0.2801324129104614, 0.37019863724708557], [0.3674626648426056, 0.46266674995422363, 0.048728037625551224, 0.3158264458179474], [0.3900073766708374, 0.2156796008348465, 0.14067278802394867, 0.45257124304771423]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_dc38fbca4d28435e70b9ea4f76b0680d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0ee0a4d6f97b1bb2fa8dff75f33b69c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.18423700332641602]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.31702283024787903]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_e2b551d16956f6458e1db91bc4f11a38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2484549731016159]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.4279927611351013]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_a0d6fe13fca6b933cd48d3de2e48a030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20194382965564728]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.31702283024787903]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_d07fa31371d130ff77cda409fc1b97ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2484549731016159]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.22766174376010895]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_834526747b8cc1dc2221d35662105e9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.18423700332641602]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.08592501282691956]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_93ad1c1b92c5e8dca9f1803706f4d172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2613685429096222]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.4279927611351013]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_9bdc6d26973b6f55a58286d7ce882acc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.01877402327954769]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_50e77b8bb8092633326e0c78177a92b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20194382965564728]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.08592501282691956]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_dd0c76a99dabff1c08c2ad49ff91765e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2613685429096222]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.22766174376010895]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_704dcd4cc892fa32a7abdcae43e9c42d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.003910623025149107]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.01877402327954769]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_1c7e53d8b655264dd403b16ea46aed9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[5.800775527954102]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_89c8b3c4d11dc1888e7bb2d65978a5c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1737920194864273], [0.04220198094844818], [0.059069257229566574], [0.08161171525716782], [0.2019866406917572], [0.0939614474773407]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.2295941263437271], [0.34228232502937317], [0.27764323353767395], [0.2966075837612152], [0.2267504632472992], [0.06496386975049973]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_34095b7369f51124e793e5cda60c1863(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0316222719848156], [0.246535524725914], [0.4039975106716156], [0.05371994525194168], [0.1470690220594406], [0.23315942287445068]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3855139911174774], [0.47146061062812805], [0.30431702733039856], [0.289742648601532], [0.24141444265842438], [0.44625771045684814]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_767d21980809a9d0ed4ed49fcb909fa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1737920194864273], [0.04220198094844818], [0.4123152494430542], [0.2897668480873108], [0.47753188014030457], [0.0939614474773407]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.21024076640605927], [0.28957539796829224], [0.1927700936794281], [0.2966075837612152], [0.0682995542883873], [0.04952847585082054]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_4e7a9e0662b2e9d65f77990bd6087c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0316222719848156], [0.3478626012802124], [0.4299014210700989], [0.05371994525194168], [0.1470690220594406], [0.3986620306968689]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.25873905420303345], [0.47146061062812805], [0.007305239327251911], [0.09287451207637787], [0.24141444265842438], [0.4163098633289337]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_0e051fc182fbd0474196fea1e9c24313(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.28047975897789], [0.05378994718194008], [0.059069257229566574], [0.08161171525716782], [0.2019866406917572], [0.4812576174736023]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.2295941263437271], [0.34228232502937317], [0.27764323353767395], [0.19415970146656036], [0.2267504632472992], [0.06496386975049973]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_bf8f0bce65f404b5da537721d8b35cad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0392889678478241], [0.246535524725914], [0.4039975106716156], [0.49785470962524414], [0.2619699537754059], [0.23315942287445068]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3855139911174774], [0.164891317486763], [0.30431702733039856], [0.289742648601532], [0.23264625668525696], [0.44625771045684814]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_ecdf299c1d05d212ae45107ee008443c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.009339757263660431], [0.007021130993962288], [0.0709913820028305], [-0.023154746741056442], [-0.03933536261320114], [-0.08949562907218933]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_7c42ac1885759587268077420f633097(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.28047975897789], [0.05378994718194008], [0.4123152494430542], [0.2897668480873108], [0.47753188014030457], [0.4812576174736023]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.21024076640605927], [0.28957539796829224], [0.1927700936794281], [0.19415970146656036], [0.0682995542883873], [0.04952847585082054]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_f933aec4a1356d86358fca2548fff71b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0392889678478241], [0.3478626012802124], [0.4299014210700989], [0.49785470962524414], [0.2619699537754059], [0.3986620306968689]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.25873905420303345], [0.164891317486763], [0.007305239327251911], [0.09287451207637787], [0.23264625668525696], [0.4163098633289337]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_8f5e3b6df42042428769001c0d60eb03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.015413952991366386], [-0.04314196854829788], [0.09277894347906113], [0.038718998432159424], [0.012000204995274544], [-0.007619083393365145]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.009339757263660431], [0.007021130993962288], [0.0709913820028305], [-0.023154746741056442], [-0.03933536261320114], [-0.08949562907218933]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_6f8a9c1a453e469cc7095146fc4b12c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [0.0], [0.0], [-0.0], [-0.0], [-0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3940712511539459], [1.1627447605133057], [0.23483304679393768], [1.598020315170288], [4.277890682220459], [-10.746246337890625]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_34873fabf1582e8f53ddb979e679adc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19651128351688385, 0.28707900643348694, 0.25880166888237, 0.06720683723688126], [0.2174926996231079, 0.2727186977863312, 0.1463117152452469, 0.2791464924812317], [0.31909412145614624, 0.17655104398727417, 0.2616613507270813, 0.037769656628370285], [0.07266577333211899, 0.2915841341018677, 0.36097460985183716, 0.27452218532562256]], dtype='float32').reshape([4, 4]),
            paddle.to_tensor([[0.43332943320274353, 0.32061561942100525, 0.27557945251464844, 0.347639262676239], [0.17252875864505768, 0.1766331046819687, 0.02188757434487343, 0.4731476306915283], [0.3685610890388489, 0.40802261233329773, 0.05652021989226341, 0.019085915759205818], [0.013710379600524902, 0.09284144639968872, 0.20442309975624084, 0.43518316745758057]], dtype='float32').reshape([4, 4]),
        ]


class TestPrimitiveOp_146a165acf7c715aa0314a1892c59985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_146a165acf7c715aa0314a1892c59985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_146a165acf7c715aa0314a1892c59985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_146a165acf7c715aa0314a1892c59985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_146a165acf7c715aa0314a1892c59985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_146a165acf7c715aa0314a1892c59985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_146a165acf7c715aa0314a1892c59985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4a112d08f8eecb192e097cac8b4a275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4a112d08f8eecb192e097cac8b4a275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4a112d08f8eecb192e097cac8b4a275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4a112d08f8eecb192e097cac8b4a275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4a112d08f8eecb192e097cac8b4a275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4a112d08f8eecb192e097cac8b4a275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4a112d08f8eecb192e097cac8b4a275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_893a1f1b15e5a74fb3569cb92d8f0544(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92cb32ccfa690f41100506ec3bb2144e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d76240a92115eb2cebd7e290b025984e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d76240a92115eb2cebd7e290b025984e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d76240a92115eb2cebd7e290b025984e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d76240a92115eb2cebd7e290b025984e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d76240a92115eb2cebd7e290b025984e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d76240a92115eb2cebd7e290b025984e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d76240a92115eb2cebd7e290b025984e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d76240a92115eb2cebd7e290b025984e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d76240a92115eb2cebd7e290b025984e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d76240a92115eb2cebd7e290b025984e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d76240a92115eb2cebd7e290b025984e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3de256f28dfb65637024ac57896c20f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26374218f165833153f20543fd3cfd0b
    def get_inputs(self):
        return [
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e32c662f8a03b3db9565f6e7f6713751(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e550e076a243ef110a84edc2744e886e
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92cb32ccfa690f41100506ec3bb2144e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_140047c5b71dc71db546c21426a5df6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.27448874711990356, 0.4337809085845947, 0.012523694895207882, 0.3010084331035614], [0.27448874711990356, 0.4337809085845947, 0.012523694895207882, 0.3010084331035614], [0.46631962060928345, 0.3336809277534485, 0.3420472741127014, 0.27878338098526], [0.47091081738471985, 0.4481792449951172, 0.21589210629463196, 0.4998915195465088], [0.3268766403198242, 0.47341400384902954, 0.28678902983665466, 0.30705517530441284], [0.009531594812870026, 0.48314884305000305, 0.02810022421181202, 0.11470696330070496], [0.18108704686164856, 0.4110396206378937, 0.1819726526737213, 0.42584389448165894]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.32735854387283325, 0.2978362441062927, 0.21544580161571503, 0.04564615339040756], [0.32735854387283325, 0.2978362441062927, 0.21544580161571503, 0.04564615339040756], [0.30271100997924805, 0.3037857413291931, 0.46283915638923645, 0.056603819131851196], [0.2832089960575104, 0.29177767038345337, 0.2168172001838684, 0.27526164054870605], [0.19069726765155792, 0.3373091220855713, 0.18438208103179932, 0.4422700107097626], [0.43759599328041077, 0.3744048476219177, 0.48488423228263855, 0.1273631453514099], [0.44247642159461975, 0.4852941930294037, 0.15796975791454315, 0.12672944366931915]], dtype='float32').reshape([7, 4]),
        ]


class TestPrimitiveOp_2898b27083759daf9c51a25aea6e85c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2898b27083759daf9c51a25aea6e85c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2898b27083759daf9c51a25aea6e85c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2898b27083759daf9c51a25aea6e85c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2898b27083759daf9c51a25aea6e85c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2898b27083759daf9c51a25aea6e85c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2898b27083759daf9c51a25aea6e85c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_63f1caead9646bf59ae45283439912eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60a8d5dc90b6d63413096f3570b268e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d17218c50c61309e237e3456e91e44b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ce515902c5e5c466f874f22d9870514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([4618, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_619f903c7f4537dbf6660913d3fe1909(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_619f903c7f4537dbf6660913d3fe1909(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_619f903c7f4537dbf6660913d3fe1909(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_619f903c7f4537dbf6660913d3fe1909(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_619f903c7f4537dbf6660913d3fe1909(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_619f903c7f4537dbf6660913d3fe1909(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_619f903c7f4537dbf6660913d3fe1909(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_619f903c7f4537dbf6660913d3fe1909(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_619f903c7f4537dbf6660913d3fe1909(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_619f903c7f4537dbf6660913d3fe1909(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_619f903c7f4537dbf6660913d3fe1909(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_77885dbde4816ce70b7b228191368af4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26374218f165833153f20543fd3cfd0b
    def get_inputs(self):
        return [
            paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cee8d4945a5b3cec7d00b42bdf15b56b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e550e076a243ef110a84edc2744e886e
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ce515902c5e5c466f874f22d9870514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([4618, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_641d0e83580804f020fa0396eac7b5e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1058, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28c44aa8a8d38d41fabc4a1ce7bfcd24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28c44aa8a8d38d41fabc4a1ce7bfcd24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28c44aa8a8d38d41fabc4a1ce7bfcd24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28c44aa8a8d38d41fabc4a1ce7bfcd24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28c44aa8a8d38d41fabc4a1ce7bfcd24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28c44aa8a8d38d41fabc4a1ce7bfcd24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28c44aa8a8d38d41fabc4a1ce7bfcd24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28c44aa8a8d38d41fabc4a1ce7bfcd24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28c44aa8a8d38d41fabc4a1ce7bfcd24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28c44aa8a8d38d41fabc4a1ce7bfcd24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28c44aa8a8d38d41fabc4a1ce7bfcd24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b76fd2d698671462d784c4e971e4ed5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26374218f165833153f20543fd3cfd0b
    def get_inputs(self):
        return [
            paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_77e193fc054f3a30c5ba139e85b19640(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e550e076a243ef110a84edc2744e886e
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_641d0e83580804f020fa0396eac7b5e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1058, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c9460cd58abff38d75364249aa3e12c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a34133090430af21ba074c987d7ba16a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1420971155166626, 0.14499685168266296, 0.03446013480424881, 0.3483777344226837], [0.3927956223487854, 0.05707862973213196, 0.43702882528305054, 0.3344132900238037], [0.3927956223487854, 0.05707862973213196, 0.43702882528305054, 0.3344132900238037], [0.2871904671192169, 0.37454575300216675, 0.04248092323541641, 0.44454556703567505], [0.14685574173927307, 0.25309816002845764, 0.48754656314849854, 0.49146023392677307], [0.16142979264259338, 0.10061225295066833, 0.0665721446275711, 0.37526777386665344]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.0107638044282794, 0.060170259326696396, 0.29249492287635803, 0.44656503200531006], [0.29241108894348145, 0.010801371186971664, 0.2764958143234253, 0.35110828280448914], [0.29241108894348145, 0.010801371186971664, 0.2764958143234253, 0.35110828280448914], [0.2598353326320648, 0.37688690423965454, 0.15778756141662598, 0.017685389146208763], [0.1843077689409256, 0.39407601952552795, 0.1135324239730835, 0.0714903473854065], [0.035217661410570145, 0.2196827381849289, 0.46750006079673767, 0.07889445126056671]], dtype='float32').reshape([6, 4]),
        ]


class TestPrimitiveOp_30bb6fd5af91ce7657d492a7e1cae8c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30bb6fd5af91ce7657d492a7e1cae8c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30bb6fd5af91ce7657d492a7e1cae8c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30bb6fd5af91ce7657d492a7e1cae8c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30bb6fd5af91ce7657d492a7e1cae8c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30bb6fd5af91ce7657d492a7e1cae8c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30bb6fd5af91ce7657d492a7e1cae8c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5f4863710bd5458a94509ce62acde5f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 2], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_47492f828e721b67b39f9215d060228c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 1, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1fc579d64cfe1e42afc1331d0c39ce0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47492f828e721b67b39f9215d060228c
    def get_inputs(self):
        return [
            paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.35187867283821106, 0.8563032150268555, 0.14742174744606018, 0.051911212503910065], [4.772407054901123, 0.36982136964797974, 0.9756341576576233, 1.5479004383087158]], dtype='float32').reshape([2, 4]),
        ]


class TestPrimitiveOp_36ab29e79a434a6f89b8806af898711a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36ab29e79a434a6f89b8806af898711a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36ab29e79a434a6f89b8806af898711a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36ab29e79a434a6f89b8806af898711a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36ab29e79a434a6f89b8806af898711a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36ab29e79a434a6f89b8806af898711a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36ab29e79a434a6f89b8806af898711a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b645ec3c5d75c4c201f7def317478b92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6069, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c0188687fe42bc70249aadb5c07d01b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b645ec3c5d75c4c201f7def317478b92
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf76ec6f6e78b8cf22c7a8c31362c408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([300, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 2], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6ff0558d29e4c36c82c5fce402c13bdb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 1, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_25098f510c1631832a9fee05942b0428(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ff0558d29e4c36c82c5fce402c13bdb
    def get_inputs(self):
        return [
            paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1.4216327667236328, 0.09512367099523544, 2.08686900138855, 0.07152839004993439], [5.2525715827941895, 1.1873953342437744, 0.880722165107727, 0.6808139681816101]], dtype='float32').reshape([2, 4]),
        ]


class TestPrimitiveOp_8657710c87cff79f742d650d62f78888(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.016903966665267944], [0.2075682431459427], [0.0038826088421046734], [0.21522128582000732], [0.0541519820690155]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.4782654941082001], [0.2401391565799713], [0.48159608244895935], [0.43455803394317627], [0.4149826467037201]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_3aaa59b55f846a6d4eb95d92b6f61510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22383494675159454], [0.38703131675720215], [0.22455859184265137], [0.28253594040870667], [0.07010837644338608]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.3702232539653778], [0.376865416765213], [0.47763171792030334], [0.4847533106803894], [0.291609525680542]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_3bdb84f1a761619405528c8fc2b5a544(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20464277267456055], [0.3423576056957245], [0.0038826088421046734], [0.21522128582000732], [0.0541519820690155]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.4358735978603363], [0.2401391565799713], [0.48159608244895935], [0.43455803394317627], [0.0802948921918869]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_6fd254d63dda0aa425bca50e99d9c510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.45856836438179016], [0.38703131675720215], [0.40157803893089294], [0.4099683463573456], [0.07010837644338608]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.10876166075468063], [0.376865416765213], [0.34192609786987305], [0.4847533106803894], [0.291609525680542]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_8e8cbea134aac97a7070f7ce5a242a54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.016903966665267944], [0.2075682431459427], [0.20523563027381897], [0.4772803485393524], [0.2485220581293106]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.4782654941082001], [0.15857505798339844], [0.3338254690170288], [0.004150536842644215], [0.4149826467037201]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_82f360a5d0ea60a16b3ad0ac58950f4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22383494675159454], [0.43285518884658813], [0.22455859184265137], [0.28253594040870667], [0.3625085949897766]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.3702232539653778], [0.3718237280845642], [0.47763171792030334], [0.08214692771434784], [0.16273191571235657]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_ca50d270d65bfffa110a8d38eb101985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.013348154723644257], [0.0040292683988809586], [0.004046095535159111], [0.11121310293674469], [-0.027464259415864944]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_155f6c29270683ff384803e7f18a6b52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20464277267456055], [0.3423576056957245], [0.20523563027381897], [0.4772803485393524], [0.2485220581293106]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.4358735978603363], [0.15857505798339844], [0.3338254690170288], [0.004150536842644215], [0.0802948921918869]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_cdd0110ae3abbd179dc4c927e5b1c80a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.45856836438179016], [0.43285518884658813], [0.40157803893089294], [0.4099683463573456], [0.3625085949897766]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.10876166075468063], [0.3718237280845642], [0.34192609786987305], [0.08214692771434784], [0.16273191571235657]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_ee5bdf3a06a001778cc836b872ea005c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.08088608831167221], [0.01121651753783226], [-0.007670633494853973], [0.15510208904743195], [0.03360786288976669]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.013348154723644257], [0.0040292683988809586], [0.004046095535159111], [0.11121310293674469], [-0.027464259415864944]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_a6fdff29ed65f7732e1547e6c56163af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [0.0], [0.0], [0.0], [-0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.8349758982658386], [0.6407736539840698], [1.5274786949157715], [0.2829683721065521], [1.8171974420547485]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_01a68e3eb8cb88ca6eaf67199755facd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d17218c50c61309e237e3456e91e44b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99d04deefd3715f422cc68ba7061f3c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99d04deefd3715f422cc68ba7061f3c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99d04deefd3715f422cc68ba7061f3c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99d04deefd3715f422cc68ba7061f3c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99d04deefd3715f422cc68ba7061f3c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99d04deefd3715f422cc68ba7061f3c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99d04deefd3715f422cc68ba7061f3c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17722cd741deeb0cdbf8fe85b557172c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17722cd741deeb0cdbf8fe85b557172c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17722cd741deeb0cdbf8fe85b557172c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17722cd741deeb0cdbf8fe85b557172c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17722cd741deeb0cdbf8fe85b557172c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17722cd741deeb0cdbf8fe85b557172c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17722cd741deeb0cdbf8fe85b557172c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_201e1e7eebf3908751515de2292d03fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_201e1e7eebf3908751515de2292d03fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_201e1e7eebf3908751515de2292d03fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_201e1e7eebf3908751515de2292d03fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_201e1e7eebf3908751515de2292d03fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_201e1e7eebf3908751515de2292d03fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_201e1e7eebf3908751515de2292d03fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_016aeea65776cb39bcc276e5ce50b2a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2402, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2af51aba37f70e343ab4f2bfb6afdaec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2af51aba37f70e343ab4f2bfb6afdaec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2af51aba37f70e343ab4f2bfb6afdaec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2af51aba37f70e343ab4f2bfb6afdaec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2af51aba37f70e343ab4f2bfb6afdaec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2af51aba37f70e343ab4f2bfb6afdaec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2af51aba37f70e343ab4f2bfb6afdaec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2af51aba37f70e343ab4f2bfb6afdaec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2af51aba37f70e343ab4f2bfb6afdaec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2af51aba37f70e343ab4f2bfb6afdaec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2af51aba37f70e343ab4f2bfb6afdaec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_24088b22d325232280317bca42ecc219(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26374218f165833153f20543fd3cfd0b
    def get_inputs(self):
        return [
            paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c17b953ff4a3815803abcde9d22f85e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e550e076a243ef110a84edc2744e886e
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_016aeea65776cb39bcc276e5ce50b2a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2402, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a46709cb12eb1a8bc3b351459326eb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5173203ef92a5ff1d38a6ac975aba67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5173203ef92a5ff1d38a6ac975aba67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5173203ef92a5ff1d38a6ac975aba67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5173203ef92a5ff1d38a6ac975aba67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5173203ef92a5ff1d38a6ac975aba67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5173203ef92a5ff1d38a6ac975aba67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5173203ef92a5ff1d38a6ac975aba67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5173203ef92a5ff1d38a6ac975aba67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5173203ef92a5ff1d38a6ac975aba67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5173203ef92a5ff1d38a6ac975aba67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5173203ef92a5ff1d38a6ac975aba67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_218e8ab77434e49b081755a5c399a0b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26374218f165833153f20543fd3cfd0b
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b00789947fbe9fc24f4956d8774b6846(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e550e076a243ef110a84edc2744e886e
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a46709cb12eb1a8bc3b351459326eb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_785b8e911c8a6330e669cea560908b0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1646e42feb6db90f608ae8396a4d5a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1646e42feb6db90f608ae8396a4d5a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1646e42feb6db90f608ae8396a4d5a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1646e42feb6db90f608ae8396a4d5a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1646e42feb6db90f608ae8396a4d5a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1646e42feb6db90f608ae8396a4d5a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1646e42feb6db90f608ae8396a4d5a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1646e42feb6db90f608ae8396a4d5a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1646e42feb6db90f608ae8396a4d5a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1646e42feb6db90f608ae8396a4d5a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1646e42feb6db90f608ae8396a4d5a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d02a8d8766fc313e871173c84524bb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26374218f165833153f20543fd3cfd0b
    def get_inputs(self):
        return [
            paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66ebd06667736dbc1497a3771ac27e20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e550e076a243ef110a84edc2744e886e
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_785b8e911c8a6330e669cea560908b0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e4b073bb59d5e61352f073488f92153(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d17218c50c61309e237e3456e91e44b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4acbe9066d4ff318767c9cc63f2c7a15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2898b27083759daf9c51a25aea6e85c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2898b27083759daf9c51a25aea6e85c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2898b27083759daf9c51a25aea6e85c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2898b27083759daf9c51a25aea6e85c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2898b27083759daf9c51a25aea6e85c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2898b27083759daf9c51a25aea6e85c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2898b27083759daf9c51a25aea6e85c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e5419054ba789c452f6abf3e6083f808(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 512, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_16e0b7154f59ad17fc196198187a2852(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5419054ba789c452f6abf3e6083f808
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64828ad41fa5af617e64d31a6e5ca9e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
            paddle.to_tensor([0.28501829504966736, 0.46016451716423035, 0.42440566420555115, 0.06916454434394836, 0.00568016804754734, 0.24515274167060852, 0.0075411563739180565, 0.3134956955909729, 0.3894185721874237, 0.2804737091064453, 0.44300535321235657, 0.23875261843204498, 0.03306839242577553, 0.20148731768131256, 0.03598868101835251, 0.04405199736356735, 0.2430863231420517, 0.007826716639101505, 0.22667871415615082, 0.35204222798347473], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_212509356fa35c66721655229316521e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.28501829504966736, 0.46016451716423035, 0.42440566420555115, 0.06916454434394836, 0.00568016804754734, 0.24515274167060852, 0.0075411563739180565, 0.3134956955909729, 0.3894185721874237, 0.2804737091064453, 0.44300535321235657, 0.23875261843204498, 0.03306839242577553, 0.20148731768131256, 0.03598868101835251, 0.04405199736356735, 0.2430863231420517, 0.007826716639101505, 0.22667871415615082, 0.35204222798347473], dtype='float32').reshape([20]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_be8eb760e3aec2725477927d9a767971(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33608508110046387], [0.17682288587093353], [0.08292681723833084], [0.04139469563961029]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.16749481856822968], [0.4973313510417938], [0.4870723485946655], [0.49783867597579956]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_9aef13f0859c0248ca480be68ad6a17f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2787860631942749], [0.025012454017996788], [0.35968491435050964], [0.11074841767549515]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.46310076117515564], [0.3204939365386963], [0.1550406515598297], [0.28060683608055115]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_280e707a19234590b906eb2b6dd05bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33608508110046387], [0.36661604046821594], [0.08292681723833084], [0.2736373543739319]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.12751974165439606], [0.4973313510417938], [0.4870723485946655], [0.49783867597579956]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_3dcd3243226b24aa1a9dda6d9f1a6735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2787860631942749], [0.025012454017996788], [0.35968491435050964], [0.26496902108192444]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.46310076117515564], [0.3204939365386963], [0.1550406515598297], [0.08015791326761246]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_f90ba5d16d13db71d96cfaf1f06a5cd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.38861891627311707], [0.17682288587093353], [0.2886216640472412], [0.04139469563961029]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.16749481856822968], [0.22387433052062988], [0.30820056796073914], [0.2222341001033783]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_d44d78ee8f5dd9359c1c0a7871346b08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4298492968082428], [0.2742873728275299], [0.4489744007587433], [0.11074841767549515]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4111035466194153], [0.13314859569072723], [0.12051653861999512], [0.28060683608055115]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_a91492aa7bba2b4285761ce47158a0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.03429652005434036], [0.03198316693305969], [-0.08913690596818924], [-0.01071779802441597]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_a2baf2e7220953c2e5731a2e8a2bbc52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.38861891627311707], [0.36661604046821594], [0.2886216640472412], [0.2736373543739319]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.12751974165439606], [0.22387433052062988], [0.30820056796073914], [0.2222341001033783]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_006c3c02068fd8b597502e3539378577(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4298492968082428], [0.2742873728275299], [0.4489744007587433], [0.26496902108192444]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4111035466194153], [0.13314859569072723], [0.12051653861999512], [0.08015791326761246]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_7ec9c7c0b04de9905ab8c912a46cea3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.00489449966698885], [0.020146390423178673], [-0.006430844776332378], [0.009499892592430115]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.03429652005434036], [0.03198316693305969], [-0.08913690596818924], [-0.01071779802441597]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_c4b783d63f5cdb4e9a251728d3c6fede(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [0.0], [-0.0], [-0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[8.007155418395996], [-0.5875383019447327], [-12.86083984375], [2.128201961517334]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_78a96c00e7d75950e6e509e9740cb87f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_625a74f45737fa0ea8ce74dfec23acd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2114, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45c2abc97bb18619ec7f5855c71344e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45c2abc97bb18619ec7f5855c71344e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45c2abc97bb18619ec7f5855c71344e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45c2abc97bb18619ec7f5855c71344e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45c2abc97bb18619ec7f5855c71344e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45c2abc97bb18619ec7f5855c71344e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45c2abc97bb18619ec7f5855c71344e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45c2abc97bb18619ec7f5855c71344e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45c2abc97bb18619ec7f5855c71344e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45c2abc97bb18619ec7f5855c71344e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45c2abc97bb18619ec7f5855c71344e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3de256f28dfb65637024ac57896c20f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26374218f165833153f20543fd3cfd0b
    def get_inputs(self):
        return [
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e32c662f8a03b3db9565f6e7f6713751(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e550e076a243ef110a84edc2744e886e
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_625a74f45737fa0ea8ce74dfec23acd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2114, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_16e0b7154f59ad17fc196198187a2852(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5419054ba789c452f6abf3e6083f808
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_abadecd7228d0eb97414638085cc336d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d17218c50c61309e237e3456e91e44b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2611ccfb3e1440e3fa428e360d3f627b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6804, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e697136b8ca7f221ad379c4e00262cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2611ccfb3e1440e3fa428e360d3f627b
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d4154a71660526963733b3936f8f8241(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.048502251505851746, 0.4599551856517792, 0.35313335061073303, 0.33025050163269043], [0.4256770610809326, 0.31103983521461487, 0.3996395170688629, 0.2820717692375183], [0.4386492073535919, 0.05111386626958847, 0.15086457133293152, 0.4465380609035492], [0.4386492073535919, 0.05111386626958847, 0.15086457133293152, 0.4465380609035492], [0.4329345226287842, 0.13714048266410828, 0.4314191937446594, 0.00365569069981575]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.03188830241560936, 0.10413093864917755, 0.43004634976387024, 0.18256761133670807], [0.3873386085033417, 0.020876504480838776, 0.13552509248256683, 0.4197688400745392], [0.4265201985836029, 0.23856724798679352, 0.29972225427627563, 0.2599564492702484], [0.4265201985836029, 0.23856724798679352, 0.29972225427627563, 0.2599564492702484], [0.15383942425251007, 0.1514587104320526, 0.29892706871032715, 0.15141336619853973]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_201e1e7eebf3908751515de2292d03fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_201e1e7eebf3908751515de2292d03fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_201e1e7eebf3908751515de2292d03fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_201e1e7eebf3908751515de2292d03fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_201e1e7eebf3908751515de2292d03fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_201e1e7eebf3908751515de2292d03fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_201e1e7eebf3908751515de2292d03fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78fb6ccbd33d2bd15825b9b21e3aecc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78fb6ccbd33d2bd15825b9b21e3aecc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78fb6ccbd33d2bd15825b9b21e3aecc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78fb6ccbd33d2bd15825b9b21e3aecc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78fb6ccbd33d2bd15825b9b21e3aecc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78fb6ccbd33d2bd15825b9b21e3aecc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78fb6ccbd33d2bd15825b9b21e3aecc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f14ff1d47cb0998cb096dafe3e24f379(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c4c21ccb93f073c25690ff547fe3587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c4c21ccb93f073c25690ff547fe3587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c4c21ccb93f073c25690ff547fe3587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c4c21ccb93f073c25690ff547fe3587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c4c21ccb93f073c25690ff547fe3587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c4c21ccb93f073c25690ff547fe3587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c4c21ccb93f073c25690ff547fe3587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_97f8686bde4c1c1bf2833e40e5d14cc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([4156, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd3014d2f4700b8a01acde4ab5e439c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd3014d2f4700b8a01acde4ab5e439c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd3014d2f4700b8a01acde4ab5e439c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd3014d2f4700b8a01acde4ab5e439c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd3014d2f4700b8a01acde4ab5e439c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd3014d2f4700b8a01acde4ab5e439c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd3014d2f4700b8a01acde4ab5e439c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd3014d2f4700b8a01acde4ab5e439c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd3014d2f4700b8a01acde4ab5e439c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd3014d2f4700b8a01acde4ab5e439c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd3014d2f4700b8a01acde4ab5e439c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_769b8a9af521d0059c28b85712444165(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26374218f165833153f20543fd3cfd0b
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_496f7835f811ba3e1142e94e479af8f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e550e076a243ef110a84edc2744e886e
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_97f8686bde4c1c1bf2833e40e5d14cc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([4156, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc0f5aa75e731779a2a2b98479b5fabc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.13482677936553955, 0.15049958229064941, 0.077986940741539, 0.12898637354373932], [0.2697294354438782, 0.01579047553241253, 0.24616752564907074, 0.0022995120380073786], [0.07843600958585739, 0.19671979546546936, 0.4115786552429199, 0.05821404233574867], [0.13482677936553955, 0.15049958229064941, 0.077986940741539, 0.12898637354373932], [0.15283596515655518, 0.3291408121585846, 0.07469618320465088, 0.1810348629951477], [0.1273183673620224, 0.11259562522172928, 0.27176937460899353, 0.08801460266113281], [0.15283596515655518, 0.3291408121585846, 0.07469618320465088, 0.1810348629951477]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.26715365052223206, 0.38852807879447937, 0.4112016260623932, 0.013165361247956753], [0.256456196308136, 0.015323239378631115, 0.2650648057460785, 0.39667803049087524], [0.24438638985157013, 0.2001534104347229, 0.176560178399086, 0.17187924683094025], [0.26715365052223206, 0.38852807879447937, 0.4112016260623932, 0.013165361247956753], [0.2117203176021576, 0.1721329241991043, 0.23883982002735138, 0.22342772781848907], [0.3143860399723053, 0.03525630384683609, 0.17414899170398712, 0.14876626431941986], [0.2117203176021576, 0.1721329241991043, 0.23883982002735138, 0.22342772781848907]], dtype='float32').reshape([7, 4]),
        ]


class TestPrimitiveOp_6c4c21ccb93f073c25690ff547fe3587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c4c21ccb93f073c25690ff547fe3587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c4c21ccb93f073c25690ff547fe3587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c4c21ccb93f073c25690ff547fe3587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c4c21ccb93f073c25690ff547fe3587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c4c21ccb93f073c25690ff547fe3587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c4c21ccb93f073c25690ff547fe3587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adcea79a7d32530c808b3fbffdecda65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f19e427e19d61cf798244fa715f82ad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()