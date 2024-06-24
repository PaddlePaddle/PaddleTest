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


class TestPrimitiveOp_c2328c1790d02b901fc64647f2aee9a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.11615855246782303]], [[0.2003442347049713]], [[0.17741313576698303]], [[0.37048137187957764]], [[0.4087255299091339]], [[0.4362022280693054]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6920855045318604]], [[0.5404301285743713]], [[0.6636646389961243]], [[0.6878827214241028]], [[0.7699432373046875]], [[0.5830170512199402]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_d2f5563545770924aa3e5a3bbc2adea5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.36727383732795715]], [[0.4077630341053009]], [[0.03247654065489769]], [[0.4937010407447815]], [[0.1328447014093399]], [[0.18904006481170654]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6160444021224976]], [[0.6274053454399109]], [[0.7407793998718262]], [[0.7812472581863403]], [[0.5053333640098572]], [[0.6351003050804138]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_2876b356ae07a13b1ad4a091178a9819(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
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


class TestPrimitiveOp_b60aa18e8413702d572a04e7953ed617(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.36989566683769226, 0.020481908693909645]], [[0.019737232476472855, 0.3435409367084503]], [[0.4357619285583496, 0.4986684322357178]], [[0.38670074939727783, 0.09431659430265427]], [[0.040527280420064926, 0.07723013311624527]], [[0.39266401529312134, 0.4662638306617737]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.27006879448890686, 0.2766728699207306]], [[0.24770629405975342, 0.12015974521636963]], [[0.09283237159252167, 0.2569616138935089]], [[0.36232709884643555, 0.1261819750070572]], [[0.10638056695461273, 0.4779474139213562]], [[0.0906803086400032, 0.02180427685379982]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_1606123393ca596208e10b7521cea4e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.36276569962501526, 0.4241918623447418]], [[0.2027422934770584, 0.3765113651752472]], [[0.3348807692527771, 0.22022031247615814]], [[0.31425759196281433, 0.41133788228034973]], [[0.43883058428764343, 0.012980783358216286]], [[0.08060404658317566, 0.17904053628444672]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.27006879448890686, 0.2766728699207306]], [[0.24770629405975342, 0.12015974521636963]], [[0.09283237159252167, 0.2569616138935089]], [[0.36232709884643555, 0.1261819750070572]], [[0.10638056695461273, 0.4779474139213562]], [[0.0906803086400032, 0.02180427685379982]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_7b360b1b8286b8ef6b05448c0548b099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3121618628501892, 0.14701275527477264]], [[0.4195224940776825, 0.38692712783813477]], [[0.38966456055641174, 0.0181307103484869]], [[0.49889588356018066, 0.46952900290489197]], [[0.49599751830101013, 0.3641703426837921]], [[0.42518582940101624, 0.1771775335073471]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class TestPrimitiveOp_3ee06cfa68dfac15840cce2c416a2963(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
            paddle.to_tensor([0.08915137499570847, 0.0019794581457972527, 0.11768212914466858, 0.4924314320087433, 0.19912375509738922, 0.1807744801044464, 0.04182153195142746, 0.18964476883411407, 0.3268153667449951, 0.22180624306201935, 0.10589847713708878, 0.03320932760834694, 0.18596151471138, 0.2120317816734314, 0.033170074224472046, 0.3353402018547058], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_7c4c7e8dac9da3069440492b6730618f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08915137499570847, 0.0019794581457972527, 0.11768212914466858, 0.4924314320087433, 0.19912375509738922, 0.1807744801044464, 0.04182153195142746, 0.18964476883411407, 0.3268153667449951, 0.22180624306201935, 0.10589847713708878, 0.03320932760834694, 0.18596151471138, 0.2120317816734314, 0.033170074224472046, 0.3353402018547058], dtype='float32').reshape([16]),
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


class TestPrimitiveOp_6933aeae73ba2696cffea9881fde98c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6933aeae73ba2696cffea9881fde98c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
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


class TestPrimitiveOp_541eb1a248367952d13cca1f363c54d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1745, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d61f595796125be0b98c0a012e729876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d61f595796125be0b98c0a012e729876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d61f595796125be0b98c0a012e729876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d61f595796125be0b98c0a012e729876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d61f595796125be0b98c0a012e729876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d61f595796125be0b98c0a012e729876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d61f595796125be0b98c0a012e729876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d61f595796125be0b98c0a012e729876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d61f595796125be0b98c0a012e729876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d61f595796125be0b98c0a012e729876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d61f595796125be0b98c0a012e729876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_541eb1a248367952d13cca1f363c54d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1745, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76c0d99f9651f6ce77e19cf7ad106b25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0765327662229538, 0.21387584507465363, 0.08543884009122849, 0.36543095111846924], [0.20723602175712585, 0.19184331595897675, 0.3878714442253113, 0.39905187487602234], [0.44137507677078247, 0.39717406034469604, 0.2068166583776474, 0.42717137932777405], [0.38834619522094727, 0.023131607100367546, 0.11142710596323013, 0.23625448346138], [0.16091035306453705, 0.24558614194393158, 0.12345148622989655, 0.0914841741323471]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.31259503960609436, 0.08323375880718231, 0.1420908123254776, 0.21166688203811646], [0.12429384142160416, 0.40907102823257446, 0.3241656720638275, 0.28161337971687317], [0.11502556502819061, 0.2864638864994049, 0.15894664824008942, 0.03576246649026871], [0.045289721339941025, 0.3379170000553131, 0.3220442533493042, 0.017203625291585922], [0.07081623375415802, 0.3748626112937927, 0.25975385308265686, 0.4770946800708771]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_8ab594189492bebff34d17b03c222636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f85507748e825f5546885f2b4c53aa37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2929908037185669, 0.3001055121421814, 0.4717049300670624, 0.3285353481769562], [0.04975004121661186, 0.028072407469153404, 0.038111258298158646, 0.28891506791114807], [0.3167937099933624, 0.2091555893421173, 0.4249999225139618, 0.12318013608455658], [0.04975004121661186, 0.028072407469153404, 0.038111258298158646, 0.28891506791114807], [0.3167937099933624, 0.2091555893421173, 0.4249999225139618, 0.12318013608455658]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.1912231296300888, 0.32807469367980957, 0.3335484266281128, 0.2526320219039917], [0.3937932848930359, 0.04521281272172928, 0.0656633973121643, 0.38378313183784485], [0.17795789241790771, 0.41247278451919556, 0.07547301054000854, 0.41287291049957275], [0.3937932848930359, 0.04521281272172928, 0.0656633973121643, 0.38378313183784485], [0.17795789241790771, 0.41247278451919556, 0.07547301054000854, 0.41287291049957275]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_4a515c1eae55579b96dc5e7fc4559d54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14627687633037567], [0.07964449375867844], [0.05605011805891991], [0.15758116543293], [0.3619658946990967], [0.35822534561157227], [0.07008229196071625], [0.24435442686080933], [0.1124526634812355]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.36898699402809143], [0.07021355628967285], [0.34596216678619385], [0.3147180676460266], [0.06824018061161041], [0.3909800946712494], [0.35480907559394836], [0.17036089301109314], [0.32742995023727417]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_686f9ca2f48c81cfbdda3d51a83328c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2213127315044403], [0.13248451054096222], [0.12742774188518524], [0.18233563005924225], [0.3280884623527527], [0.06383621692657471], [0.1773354858160019], [0.42756208777427673], [0.2447492927312851]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.1222463995218277], [0.36551961302757263], [0.3966253995895386], [0.3700411021709442], [0.20033740997314453], [0.13268493115901947], [0.20981046557426453], [0.3208114802837372], [0.22876402735710144]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_0ed863b378638ff19d0efd10cb60c166(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14627687633037567], [0.07964449375867844], [0.33320513367652893], [0.4071856439113617], [0.3619658946990967], [0.43402859568595886], [0.4941118359565735], [0.24435442686080933], [0.40847325325012207]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.30527326464653015], [0.07021355628967285], [0.34596216678619385], [0.2309519201517105], [0.05144184082746506], [0.09776482731103897], [0.29527753591537476], [0.17036089301109314], [0.15745829045772552]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_776a5852e98796072cace2e6bcd88420(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2213127315044403], [0.13248451054096222], [0.12742774188518524], [0.18233563005924225], [0.41723570227622986], [0.06383621692657471], [0.40079569816589355], [0.42756208777427673], [0.2447492927312851]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.06501298397779465], [0.03647551313042641], [0.3966253995895386], [0.3700411021709442], [0.20033740997314453], [0.13268493115901947], [0.20981046557426453], [0.3208114802837372], [0.22588470578193665]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_61aec0a8bdf79c2b9d17cd801eb9abaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19455237686634064], [0.35046592354774475], [0.05605011805891991], [0.15758116543293], [0.41687849164009094], [0.35822534561157227], [0.07008229196071625], [0.4353342056274414], [0.1124526634812355]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.36898699402809143], [0.06626627594232559], [0.12720561027526855], [0.3147180676460266], [0.06824018061161041], [0.3909800946712494], [0.35480907559394836], [0.016808612272143364], [0.32742995023727417]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_83ecbe0beb6acb0d6ead625d5ac62de2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.446295827627182], [0.16909025609493256], [0.38718894124031067], [0.32923561334609985], [0.3280884623527527], [0.4127715229988098], [0.1773354858160019], [0.4922620356082916], [0.3027809262275696]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.1222463995218277], [0.36551961302757263], [0.1967533528804779], [0.3386157155036926], [0.08081060647964478], [0.049536582082509995], [0.14338451623916626], [0.04308648407459259], [0.22876402735710144]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_a51a5a24901e1934d8d3ab0a362cf638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.08137653023004532], [-0.05491970106959343], [-0.010116374120116234], [-0.031606074422597885], [0.15356266498565674], [-0.03504899889230728], [0.028307661414146423], [0.19589032232761383], [-0.011176660656929016]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.037523772567510605], [0.0], [0.0], [0.007898855023086071], [0.0]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_760b204faf00e599a7ce4353062609e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19455237686634064], [0.35046592354774475], [0.33320513367652893], [0.4071856439113617], [0.41687849164009094], [0.43402859568595886], [0.4941118359565735], [0.4353342056274414], [0.40847325325012207]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.30527326464653015], [0.06626627594232559], [0.12720561027526855], [0.2309519201517105], [0.05144184082746506], [0.09776482731103897], [0.29527753591537476], [0.016808612272143364], [0.15745829045772552]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_c232f0c0ea798ce8aaafb8b6000be808(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.446295827627182], [0.16909025609493256], [0.38718894124031067], [0.32923561334609985], [0.41723570227622986], [0.4127715229988098], [0.40079569816589355], [0.4922620356082916], [0.3027809262275696]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.06501298397779465], [0.03647551313042641], [0.1967533528804779], [0.3386157155036926], [0.08081060647964478], [0.049536582082509995], [0.14338451623916626], [0.04308648407459259], [0.22588470578193665]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_50c1ed279173b3fb7fc4d31e2dc83c14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.042215973138809204], [0.03768906369805336], [0.03922963887453079], [-0.0016530902357771993], [0.12294206023216248], [0.12214275449514389], [0.05118217319250107], [0.18799147009849548], [0.01930209994316101]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.08137653023004532], [-0.05491970106959343], [-0.010116374120116234], [-0.031606074422597885], [0.11603888869285583], [-0.03504899889230728], [0.028307661414146423], [0.18799147009849548], [-0.011176660656929016]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_f2ebbe166e2149259202c3bbecf5e7b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [-0.0], [-0.0], [-0.0], [0.32337239384651184], [-0.0], [0.0], [0.04201709106564522], [-0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.9276241660118103], [2.4571785926818848], [1.2578758001327515], [-18.119388580322266], [0.05614979565143585], [1.2869510650634766], [0.4469234049320221], [0.0], [1.5790386199951172]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_b683a7538b123f277ffef55f48129604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_336c21da8ab6205aff5bc2d57b5e0cf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.017238300293684006]], [[0.027982017025351524]], [[0.04397109895944595]], [[0.11139567941427231]], [[0.09571146219968796]], [[0.1168811172246933]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.7047238349914551]], [[0.5973381996154785]], [[0.5072594285011292]], [[0.779000461101532]], [[0.7982144355773926]], [[0.6050887107849121]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_8eb1d9edf8e2a2d31059fb1f6efb5ea4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.008880204521119595]], [[0.2174333930015564]], [[0.2707148790359497]], [[0.4569109082221985]], [[0.04954489320516586]], [[0.06489719450473785]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.5408740043640137]], [[0.7518698573112488]], [[0.5435909628868103]], [[0.5815163254737854]], [[0.8064696788787842]], [[0.5394160747528076]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_c31382c9e3e3a476da7fe5a44630f780(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8220030541c07f63b117f40a481a78b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5556, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30490482fe3fb55bb42ca97d8067dfd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30490482fe3fb55bb42ca97d8067dfd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30490482fe3fb55bb42ca97d8067dfd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30490482fe3fb55bb42ca97d8067dfd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30490482fe3fb55bb42ca97d8067dfd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30490482fe3fb55bb42ca97d8067dfd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30490482fe3fb55bb42ca97d8067dfd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30490482fe3fb55bb42ca97d8067dfd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30490482fe3fb55bb42ca97d8067dfd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30490482fe3fb55bb42ca97d8067dfd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30490482fe3fb55bb42ca97d8067dfd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f8220030541c07f63b117f40a481a78b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5556, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1070d31c37332b3df7c99b6a105173a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.28289559483528137, 0.4845932722091675, 0.3676605820655823, 0.43051329255104065], [0.23293931782245636, 0.25469064712524414, 0.1946612447500229, 0.3098498284816742], [0.07854396849870682, 0.13543736934661865, 0.42343825101852417, 0.46920812129974365], [0.23293931782245636, 0.25469064712524414, 0.1946612447500229, 0.3098498284816742], [0.07854396849870682, 0.13543736934661865, 0.42343825101852417, 0.46920812129974365], [0.05115962401032448, 0.07877905666828156, 0.3095535337924957, 0.14990273118019104], [0.05115962401032448, 0.07877905666828156, 0.3095535337924957, 0.14990273118019104]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.13173961639404297, 0.3212623596191406, 0.36677977442741394, 0.4714911878108978], [0.16534703969955444, 0.37184953689575195, 0.19974233210086823, 0.23861533403396606], [0.15436731278896332, 0.4458079934120178, 0.0920325368642807, 0.37641671299934387], [0.16534703969955444, 0.37184953689575195, 0.19974233210086823, 0.23861533403396606], [0.15436731278896332, 0.4458079934120178, 0.0920325368642807, 0.37641671299934387], [0.11535894870758057, 0.10193796455860138, 0.34226831793785095, 0.39777296781539917], [0.11535894870758057, 0.10193796455860138, 0.34226831793785095, 0.39777296781539917]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_a2fb490d1eeb01a0cda39689f347e003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05024552345275879, 0.39770445227622986, 0.47278454899787903, 0.15071414411067963, 0.23610089719295502, 0.47662022709846497], dtype='float32').reshape([6]),
            paddle.to_tensor([0.06802684813737869, 0.3697265684604645, 0.02440202794969082, 0.18332761526107788, 0.2911740839481354, 0.12811702489852905], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_b914d5d75dd7a41eda524c15f37283df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.29055315256118774, 0.4195096492767334, 0.1857345551252365, 0.33264780044555664, 0.3535158038139343, 0.09682358801364899], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2703670859336853, 0.049813900142908096, 0.4583689868450165, 0.21031175553798676, 0.43299606442451477, 0.14105597138404846], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_b3702cfc6c1afebee9a36fe1f31e18fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10607437044382095, 0.45157837867736816, 0.14915001392364502, 0.35991302132606506, 0.15391811728477478, 0.4380766749382019], dtype='float32').reshape([6]),
            paddle.to_tensor([0.05370360612869263, 0.2583759129047394, 0.02003302238881588, 0.032925594598054886, 0.4194084107875824, 0.24436035752296448], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_05803df56627413e0dbd3f69e1a21467(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1136271134018898, 0.025614285841584206, 0.3276135325431824, 0.17587409913539886, 0.43374499678611755, 0.010036587715148926], dtype='float32').reshape([6]),
            paddle.to_tensor([0.23499614000320435, 0.061109770089387894, 0.17594373226165771, 0.2823810279369354, 0.29694291949272156, 0.009236985817551613], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_deccd4418c616fec824f8ab6685511b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06802684813737869, 0.39770445227622986, 0.14915001392364502, 0.18332761526107788, 0.15391811728477478, 0.4380766749382019], dtype='float32').reshape([6]),
            paddle.to_tensor([0.06802684813737869, 0.3697265684604645, 0.02440202794969082, 0.18332761526107788, 0.4194084107875824, 0.24436035752296448], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_675523cbd32041054719be70e39edb88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1136271134018898, 0.025614285841584206, 0.3276135325431824, 0.17587409913539886, 0.43299606442451477, 0.010036587715148926], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2703670859336853, 0.061109770089387894, 0.4583689868450165, 0.2823810279369354, 0.43299606442451477, 0.14105597138404846], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_e082177c6e48c692478bb2015b2a3066(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06802684813737869, 0.39770445227622986, 0.47278454899787903, 0.18332761526107788, 0.2911740839481354, 0.47662022709846497], dtype='float32').reshape([6]),
            paddle.to_tensor([0.06802684813737869, 0.3697265684604645, 0.02440202794969082, 0.18332761526107788, 0.2911740839481354, 0.12811702489852905], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_28ff5a94b03b05d5fd482098ad933c56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.29055315256118774, 0.4195096492767334, 0.4583689868450165, 0.33264780044555664, 0.43299606442451477, 0.14105597138404846], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2703670859336853, 0.049813900142908096, 0.4583689868450165, 0.21031175553798676, 0.43299606442451477, 0.14105597138404846], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_7cf51c93833e60eaae8bcd6c34a952d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.00635618856176734, 0.0034854901023209095, 0.019583148881793022, -0.03482642397284508, -0.03631962463259697, 0.0001548959407955408], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.0, -0.0, -0.0, -0.0, -0.0, -0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_982670efb3e26966b38174cbd40108c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05913618579506874, 0.38371551036834717, 0.2485932856798172, 0.16702088713645935, 0.2636374831199646, 0.3023686408996582], dtype='float32').reshape([6]),
            paddle.to_tensor([0.07988898456096649, 0.3549771308898926, 0.08459151536226273, 0.19641931354999542, 0.2866632640361786, 0.3412185311317444], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_209e87b1cc9691190d28c587748392a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2804601192474365, 0.2346617728471756, 0.3220517635345459, 0.2714797854423523, 0.39325594902038574, 0.11893977969884872], dtype='float32').reshape([6]),
            paddle.to_tensor([0.17431162297725677, 0.043362028896808624, 0.25177863240242004, 0.22912755608558655, 0.36534395813941956, 0.00963678676635027], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ef995bba18525dae3fec174b72fbd391(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10607437044382095, 0.45157837867736816, 0.47278454899787903, 0.35991302132606506, 0.2911740839481354, 0.47662022709846497], dtype='float32').reshape([6]),
            paddle.to_tensor([0.05370360612869263, 0.2583759129047394, 0.02003302238881588, 0.032925594598054886, 0.2911740839481354, 0.12811702489852905], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_5ec6476b420526bfcab380c144df7a4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.29055315256118774, 0.4195096492767334, 0.4583689868450165, 0.33264780044555664, 0.43374499678611755, 0.14105597138404846], dtype='float32').reshape([6]),
            paddle.to_tensor([0.23499614000320435, 0.049813900142908096, 0.17594373226165771, 0.21031175553798676, 0.29694291949272156, 0.009236985817551613], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_dddeb9ec3490f8ddc84dc956ca6d592a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.40736350417137146, -1.3891009092330933, 0.7052502036094666, -1.2559118270874023, -1.094998836517334, 1.5666687488555908], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.7221456170082092, 0.0755341500043869, -1.0244861841201782, -0.2605300843715668, 0.6059560775756836, -1.4445503950119019], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_7c808f2888186407471a1b8267dd81c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1744, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e29fe1acc80b945f0d3867bfcf082fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e29fe1acc80b945f0d3867bfcf082fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e29fe1acc80b945f0d3867bfcf082fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e29fe1acc80b945f0d3867bfcf082fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e29fe1acc80b945f0d3867bfcf082fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e29fe1acc80b945f0d3867bfcf082fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e29fe1acc80b945f0d3867bfcf082fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e29fe1acc80b945f0d3867bfcf082fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e29fe1acc80b945f0d3867bfcf082fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e29fe1acc80b945f0d3867bfcf082fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e29fe1acc80b945f0d3867bfcf082fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_7c808f2888186407471a1b8267dd81c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1744, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c46db15154ef35bed66e9dd39e5923d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b450a27346bb1e2a3fea12560ecfd7f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
            paddle.to_tensor([0.05784948170185089, 0.43556898832321167, 0.4312872588634491, 0.4016205370426178, 0.46198728680610657, 0.0058740247040987015, 0.16386902332305908, 0.3483448922634125, 0.3245091736316681, 0.36026546359062195, 0.07979327440261841, 0.02710764855146408, 0.06300970911979675, 0.109025739133358, 0.2896069288253784, 0.13761979341506958, 0.318467378616333, 0.38401126861572266, 0.43845245242118835, 0.16789230704307556, 0.10596621036529541, 0.3684060275554657, 0.28659170866012573, 0.26515650749206543], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_19e55b98aa133d572ce0ad8acafae3d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05784948170185089, 0.43556898832321167, 0.4312872588634491, 0.4016205370426178, 0.46198728680610657, 0.0058740247040987015, 0.16386902332305908, 0.3483448922634125, 0.3245091736316681, 0.36026546359062195, 0.07979327440261841, 0.02710764855146408, 0.06300970911979675, 0.109025739133358, 0.2896069288253784, 0.13761979341506958, 0.318467378616333, 0.38401126861572266, 0.43845245242118835, 0.16789230704307556, 0.10596621036529541, 0.3684060275554657, 0.28659170866012573, 0.26515650749206543], dtype='float32').reshape([24]),
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


class TestPrimitiveOp_f07ceaf2f39305058700fc20732a2175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1547, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52335553d0f3462946d23797673b0938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52335553d0f3462946d23797673b0938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52335553d0f3462946d23797673b0938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52335553d0f3462946d23797673b0938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52335553d0f3462946d23797673b0938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52335553d0f3462946d23797673b0938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52335553d0f3462946d23797673b0938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52335553d0f3462946d23797673b0938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52335553d0f3462946d23797673b0938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52335553d0f3462946d23797673b0938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52335553d0f3462946d23797673b0938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f07ceaf2f39305058700fc20732a2175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1547, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_34f060d8dc73cf4cbc0580c3045618ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
            paddle.to_tensor([0.3001556098461151, 0.17418614029884338, 0.20308345556259155, 0.2840249538421631], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_07641244d513eafaee69967de1eafa88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3001556098461151, 0.17418614029884338, 0.20308345556259155, 0.2840249538421631], dtype='float32').reshape([4]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([4]),
        ]


class PrimitiveOp_a4632ecd0ed1002eb6f2dbaaec52e6ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b97ba0ec4d6be024f3f7076c1b29831b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4632ecd0ed1002eb6f2dbaaec52e6ca
    def get_inputs(self):
        return [
            paddle.to_tensor([4], dtype='int32').reshape([1]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_ff811b0842c824e64e360a435b15acf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4632ecd0ed1002eb6f2dbaaec52e6ca
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


class TestPrimitiveOp_7060b4fc172dc863d19e8a84d4a2e4cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4110970199108124, 0.29652589559555054, 0.003945669624954462, 0.19473879039287567], [0.27747872471809387, 0.2945447862148285, 0.4189302921295166, 0.11733750253915787], [0.448369562625885, 0.3660053014755249, 0.4635598957538605, 0.39540207386016846], [0.3763394355773926, 0.11126769334077835, 0.045732807368040085, 0.21720479428768158], [0.3763394355773926, 0.11126769334077835, 0.045732807368040085, 0.21720479428768158], [0.448369562625885, 0.3660053014755249, 0.4635598957538605, 0.39540207386016846]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.44540920853614807, 0.32264047861099243, 0.31650015711784363, 0.14112059772014618], [0.006694519426673651, 0.10186109691858292, 0.4250867962837219, 0.02425203286111355], [0.3435523211956024, 0.14469484984874725, 0.19909434020519257, 0.11396010965108871], [0.1961524337530136, 0.22254282236099243, 0.4868149161338806, 0.12020273506641388], [0.1961524337530136, 0.22254282236099243, 0.4868149161338806, 0.12020273506641388], [0.3435523211956024, 0.14469484984874725, 0.19909434020519257, 0.11396010965108871]], dtype='float32').reshape([6, 4]),
        ]


class TestPrimitiveOp_2a8925b67d9809e2f66ac30dd9d23446(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0037724762223660946, 0.3166971504688263, 0.4774005711078644, 0.45093676447868347], [0.04997891187667847, 0.29293420910835266, 0.4914323687553406, 0.07889453321695328], [0.36238735914230347, 0.09018300473690033, 0.10260890424251556, 0.21478834748268127], [0.41863346099853516, 0.3260485529899597, 0.22449953854084015, 0.2069907784461975], [0.0037724762223660946, 0.3166971504688263, 0.4774005711078644, 0.45093676447868347]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.20629005134105682, 0.02234548330307007, 0.16823722422122955, 0.47007450461387634], [0.32729625701904297, 0.2573091685771942, 0.3813242018222809, 0.4093102514743805], [0.006135561503469944, 0.3066445589065552, 0.14182259142398834, 0.1139960065484047], [0.3619973659515381, 0.4229755997657776, 0.2687116861343384, 0.08100301772356033], [0.20629005134105682, 0.02234548330307007, 0.16823722422122955, 0.47007450461387634]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_dc38fbca4d28435e70b9ea4f76b0680d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3fc35ea652645412ec4a196ebd5185a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0789174735546112]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.2048177421092987]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_70eaadc4c5d548d0ea8051cf4c995133(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0012605844531208277]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.19873455166816711]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_2ac3c60719fa73bebe34bb0845506134(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.385765939950943]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.061764106154441833]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_0b81255213cc53b8330c43323f50c2a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4691363573074341]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.19873455166816711]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_e3fc35ea652645412ec4a196ebd5185a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0789174735546112]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.2048177421092987]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_c4c19449daa5fe5f28dff1dc1d95a704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0012605844531208277]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.005689022596925497]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_f737e161bb866654bea0bd1e1572c2ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08816822618246078]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_2ac3c60719fa73bebe34bb0845506134(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.385765939950943]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.061764106154441833]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_9b69b988c2a4a0a5b206c12c315464f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4691363573074341]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.005689022596925497]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_03cc3cf5bac944b377d33a2f97d03205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15015779435634613]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.08816822618246078]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_d08bc5a7da3a63d6af3188991dbd44f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.41282951831817627]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_8b296c97b8ee9e41ee92905891900644(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2444322109222412], [0.029868239536881447], [0.09312184154987335], [0.29760172963142395], [0.14579564332962036], [0.3902941942214966]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3666752576828003], [0.19270555675029755], [0.3988319933414459], [0.4834783375263214], [0.4011060297489166], [0.3782949447631836]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_0765bb19d019b97aeaa2956515c43bc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05468965321779251], [0.3557642102241516], [0.07922337204217911], [0.2927427291870117], [0.0014592537190765142], [0.0375727079808712]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.44152939319610596], [0.4197368919849396], [0.18639807403087616], [0.18250931799411774], [0.45314154028892517], [0.34045061469078064]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_ddca0dbd667a8f7ce5f063af3fbe8d1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2444322109222412], [0.05394706130027771], [0.4057570695877075], [0.33757326006889343], [0.24670080840587616], [0.3902941942214966]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3666752576828003], [0.19270555675029755], [0.09260793775320053], [0.18456070125102997], [0.4011060297489166], [0.3782949447631836]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_572334842685a90b53baf70a7fa93cc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.49187493324279785], [0.36524298787117004], [0.07922337204217911], [0.4743187427520752], [0.0014592537190765142], [0.0375727079808712]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.37516072392463684], [0.3007451593875885], [0.010893420316278934], [0.08891236782073975], [0.005648438353091478], [0.16997677087783813]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_c68b719e34ed61ddfc03e34c7cbb884c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3076860010623932], [0.029868239536881447], [0.09312184154987335], [0.29760172963142395], [0.14579564332962036], [0.46690183877944946]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0866883173584938], [0.13139678537845612], [0.3988319933414459], [0.4834783375263214], [0.05863175541162491], [0.11190173774957657]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_44490002dbb18f6f3c5f92dd0beca9e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05468965321779251], [0.3557642102241516], [0.08949944376945496], [0.2927427291870117], [0.0025271896738559008], [0.2078334242105484]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.44152939319610596], [0.4197368919849396], [0.18639807403087616], [0.18250931799411774], [0.45314154028892517], [0.34045061469078064]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_f31511d083b4ce15baf27b8730530050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.09975819289684296], [-0.0024545681662857533], [0.05102035775780678], [0.03848220407962799], [-0.038630466908216476], [-0.04866786673665047]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_295b1a71abd62578644e06a321f73a05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3076860010623932], [0.05394706130027771], [0.4057570695877075], [0.33757326006889343], [0.24670080840587616], [0.46690183877944946]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0866883173584938], [0.13139678537845612], [0.09260793775320053], [0.18456070125102997], [0.05863175541162491], [0.11190173774957657]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_1bc8abdaff9ff118e9c2bede219353a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.49187493324279785], [0.36524298787117004], [0.08949944376945496], [0.4743187427520752], [0.0025271896738559008], [0.2078334242105484]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.37516072392463684], [0.3007451593875885], [0.010893420316278934], [0.08891236782073975], [0.005648438353091478], [0.16997677087783813]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_c8d13c057e77af0bf5da11d5c2445aaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.025793571025133133], [-0.004995339084416628], [0.02461540699005127], [0.05897201597690582], [-0.0005870101158507168], [0.013439116068184376]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.09975819289684296], [-0.0024545681662857533], [0.05102035775780678], [0.03848220407962799], [-0.038630466908216476], [-0.04866786673665047]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_7c0a848c0e1f00c1798ea65f99dff881(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [-0.0], [0.0], [0.0], [-0.0], [-0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[4.867560386657715], [0.5086283087730408], [-1.0727001428604126], [0.34744974970817566], [-64.8088607788086], [4.621359348297119]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_7377e4b6c7b3567caaa3aacc6cbb19a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09048675000667572, 0.3787645995616913, 0.09583459794521332, 0.15627044439315796], [0.45783448219299316, 0.04858386516571045, 0.06434021145105362, 0.47078612446784973], [0.28694918751716614, 0.4472043812274933, 0.22057004272937775, 0.11127184331417084], [0.29785701632499695, 0.22979559004306793, 0.028952617198228836, 0.25264108180999756]], dtype='float32').reshape([4, 4]),
            paddle.to_tensor([[0.4518062174320221, 0.04336459934711456, 0.12367621809244156, 0.07663444429636002], [0.04734795540571213, 0.43793168663978577, 0.36356407403945923, 0.31604281067848206], [0.3736291527748108, 0.25109100341796875, 0.15893970429897308, 0.08816004544496536], [0.41985395550727844, 0.14290891587734222, 0.4200916290283203, 0.005338029470294714]], dtype='float32').reshape([4, 4]),
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


class TestPrimitiveOp_9bf6679c423fe07571fdd0af0c6b682b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2056, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9376a4bd874d75c4ce0812d63d6a636e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9376a4bd874d75c4ce0812d63d6a636e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9376a4bd874d75c4ce0812d63d6a636e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9376a4bd874d75c4ce0812d63d6a636e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9376a4bd874d75c4ce0812d63d6a636e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9376a4bd874d75c4ce0812d63d6a636e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9376a4bd874d75c4ce0812d63d6a636e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9376a4bd874d75c4ce0812d63d6a636e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9376a4bd874d75c4ce0812d63d6a636e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9376a4bd874d75c4ce0812d63d6a636e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9376a4bd874d75c4ce0812d63d6a636e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_9bf6679c423fe07571fdd0af0c6b682b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2056, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c522ddb36d8865387bcf17887d78ce4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3237507939338684, 0.3148842751979828, 0.19893331825733185, 0.019628487527370453], [0.3237507939338684, 0.3148842751979828, 0.19893331825733185, 0.019628487527370453], [0.006049745716154575, 0.014560168609023094, 0.36914360523223877, 0.2589817941188812], [0.3402383327484131, 0.2506375312805176, 0.18422701954841614, 0.3613077998161316], [0.01530829444527626, 0.3738483488559723, 0.27695074677467346, 0.22977906465530396], [0.01949300989508629, 0.1240517646074295, 0.3098291754722595, 0.40876075625419617], [0.22408753633499146, 0.14804741740226746, 0.1388675421476364, 0.14433294534683228]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.01187223568558693, 0.37997671961784363, 0.4928607940673828, 0.32848015427589417], [0.01187223568558693, 0.37997671961784363, 0.4928607940673828, 0.32848015427589417], [0.16995540261268616, 0.387918084859848, 0.10048455744981766, 0.043532487004995346], [0.08724136650562286, 0.24423347413539886, 0.3827553689479828, 0.2286049872636795], [0.11175305396318436, 0.47796809673309326, 0.3427310287952423, 0.15226024389266968], [0.46764758229255676, 0.20201735198497772, 0.23267662525177002, 0.3440033793449402], [0.43195298314094543, 0.37640848755836487, 0.05510794743895531, 0.06425850093364716]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_c42618cdfca35781c4b6018fbd14a78e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_507330a23b6171601a60d7b7a00ac53f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4650, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5759977d0892153bf60fa21fced725ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5759977d0892153bf60fa21fced725ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5759977d0892153bf60fa21fced725ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5759977d0892153bf60fa21fced725ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5759977d0892153bf60fa21fced725ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5759977d0892153bf60fa21fced725ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5759977d0892153bf60fa21fced725ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5759977d0892153bf60fa21fced725ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5759977d0892153bf60fa21fced725ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5759977d0892153bf60fa21fced725ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5759977d0892153bf60fa21fced725ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_507330a23b6171601a60d7b7a00ac53f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4650, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f83279ba40c206dfda04e1b193b62ca0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1059, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ff0cde16c2e92fd0c732c7ecfcc8a65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ff0cde16c2e92fd0c732c7ecfcc8a65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ff0cde16c2e92fd0c732c7ecfcc8a65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ff0cde16c2e92fd0c732c7ecfcc8a65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ff0cde16c2e92fd0c732c7ecfcc8a65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ff0cde16c2e92fd0c732c7ecfcc8a65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ff0cde16c2e92fd0c732c7ecfcc8a65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ff0cde16c2e92fd0c732c7ecfcc8a65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ff0cde16c2e92fd0c732c7ecfcc8a65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ff0cde16c2e92fd0c732c7ecfcc8a65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ff0cde16c2e92fd0c732c7ecfcc8a65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f83279ba40c206dfda04e1b193b62ca0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1059, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c9460cd58abff38d75364249aa3e12c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4e5f1905c03b0bca41876dadee93e0a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3343122601509094, 0.1868288815021515, 0.13126325607299805, 0.16395072638988495], [0.06831204891204834, 0.36682289838790894, 0.03485509008169174, 0.4793292284011841], [0.06831204891204834, 0.36682289838790894, 0.03485509008169174, 0.4793292284011841], [0.409408837556839, 0.48272979259490967, 0.20098784565925598, 0.005137384869158268], [0.43075546622276306, 0.33572399616241455, 0.07078070938587189, 0.1468321830034256], [0.15971526503562927, 0.4149702787399292, 0.2467954009771347, 0.26131874322891235]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.26057949662208557, 0.003829711116850376, 0.34465640783309937, 0.16696207225322723], [0.09719329327344894, 0.40105924010276794, 0.4727449417114258, 0.30888888239860535], [0.09719329327344894, 0.40105924010276794, 0.4727449417114258, 0.30888888239860535], [0.2765957713127136, 0.38023415207862854, 0.07674776762723923, 0.48258358240127563], [0.24248243868350983, 0.26805591583251953, 0.04857439547777176, 0.07953433692455292], [0.024453122168779373, 0.0886516347527504, 0.1461566686630249, 0.4716673195362091]], dtype='float32').reshape([6, 4]),
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


class TestPrimitiveOp_5a3434f026962688ce44285be36dd0e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e550e076a243ef110a84edc2744e886e
    def get_inputs(self):
        return [
            paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.6101153492927551, 2.500788450241089, 0.32046547532081604, 1.6472430229187012], [1.0099784135818481, 6.345240116119385, 5.326774597167969, 0.20483003556728363]], dtype='float32').reshape([2, 4]),
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


class TestPrimitiveOp_e6b3719b92a75e8ce99fe98871be80b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
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


class TestPrimitiveOp_201b89b825bb7fe6171c21baae9349e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e550e076a243ef110a84edc2744e886e
    def get_inputs(self):
        return [
            paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1.81611168384552, 0.2808106541633606, 1.4616960287094116, 36.77274703979492], [1.221671223640442, 1.1132539510726929, 0.8734368681907654, 0.7155608534812927]], dtype='float32').reshape([2, 4]),
        ]


class TestPrimitiveOp_532d87f988de2cccb996b930bb3d6d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.034823983907699585], [0.024908585473895073], [0.3305168151855469], [0.3927406668663025], [0.0862424373626709]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.40543869137763977], [0.4942517578601837], [0.45203348994255066], [0.3473803997039795], [0.16088062524795532]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_68eb3c4ecca5c76c8e6d7d02206e3e3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4671558141708374], [0.2678717076778412], [0.27585870027542114], [0.10361013561487198], [0.2139054387807846]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.34039562940597534], [0.25430768728256226], [0.45005354285240173], [0.4590291678905487], [0.3022708296775818]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_b9f22b55653a41b11f8411d26a2bcd1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.034823983907699585], [0.024908585473895073], [0.3305168151855469], [0.3927406668663025], [0.1830960065126419]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.40543869137763977], [0.47366735339164734], [0.45203348994255066], [0.09204760938882828], [0.16088062524795532]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_211bcd47814e703f37b1317480afd461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4869818389415741], [0.2678717076778412], [0.3607769310474396], [0.23778222501277924], [0.26237985491752625]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.14999118447303772], [0.25430768728256226], [0.45005354285240173], [0.4590291678905487], [0.3022708296775818]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_6266d17212628c8293907066e03ca60a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.35219553112983704], [0.37796422839164734], [0.45025119185447693], [0.43181949853897095], [0.0862424373626709]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.34268254041671753], [0.4942517578601837], [0.3020826578140259], [0.3473803997039795], [0.059936948120594025]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_7f2b3badebf6f1b2c86ef7dda7b48815(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4671558141708374], [0.34698235988616943], [0.27585870027542114], [0.10361013561487198], [0.2139054387807846]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.34039562940597534], [0.09946838766336441], [0.26270535588264465], [0.1753489077091217], [0.06309731304645538]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_ce1cf29c387be3170b558dea15f48681(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.12368782609701157], [-0.034869760274887085], [0.012797508388757706], [-0.07258497923612595], [0.003080888418480754]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_b09b369554a025f214b807c7922ed93e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.35219553112983704], [0.37796422839164734], [0.45025119185447693], [0.43181949853897095], [0.1830960065126419]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.34268254041671753], [0.47366735339164734], [0.3020826578140259], [0.09204760938882828], [0.059936948120594025]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_060ae9988be35fcd2eac316ada020630(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4869818389415741], [0.34698235988616943], [0.3607769310474396], [0.23778222501277924], [0.26237985491752625]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.14999118447303772], [0.09946838766336441], [0.26270535588264465], [0.1753489077091217], [0.06309731304645538]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_89fa2833115f126386fc4e27729a5949(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0032057890202850103], [-0.02368786185979843], [0.014531121589243412], [0.02121308632194996], [0.02454344928264618]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.12368782609701157], [-0.034869760274887085], [0.012797508388757706], [-0.07258497923612595], [0.003080888418480754]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_5bd83a7a4a92dee2573c5b0e48ef4fe9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [-0.0], [0.0], [-0.0], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[39.582645416259766], [-0.47205182909965515], [0.11930346488952637], [4.421707630157471], [0.8744720816612244]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_e49b8cf826aa2ae4730bd8c576cdfcc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
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


class TestPrimitiveOp_bf33970f8c2e82d422d12a68a1868603(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2347, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e9d0514011ca93854515f7a29d2cf36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e9d0514011ca93854515f7a29d2cf36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e9d0514011ca93854515f7a29d2cf36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e9d0514011ca93854515f7a29d2cf36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e9d0514011ca93854515f7a29d2cf36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e9d0514011ca93854515f7a29d2cf36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e9d0514011ca93854515f7a29d2cf36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e9d0514011ca93854515f7a29d2cf36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e9d0514011ca93854515f7a29d2cf36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e9d0514011ca93854515f7a29d2cf36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e9d0514011ca93854515f7a29d2cf36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_bf33970f8c2e82d422d12a68a1868603(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2347, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7742021f7bd562b5eb85f0ca7e11c133(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3109, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0dc2db4a5a4e1b6c5406b2d422f29e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0dc2db4a5a4e1b6c5406b2d422f29e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0dc2db4a5a4e1b6c5406b2d422f29e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0dc2db4a5a4e1b6c5406b2d422f29e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0dc2db4a5a4e1b6c5406b2d422f29e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0dc2db4a5a4e1b6c5406b2d422f29e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0dc2db4a5a4e1b6c5406b2d422f29e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0dc2db4a5a4e1b6c5406b2d422f29e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0dc2db4a5a4e1b6c5406b2d422f29e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0dc2db4a5a4e1b6c5406b2d422f29e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0dc2db4a5a4e1b6c5406b2d422f29e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_7742021f7bd562b5eb85f0ca7e11c133(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3109, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a99aa0d1919980cadc2fb18820335855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3813, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72826a8337ed93b7e576d245892da980(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72826a8337ed93b7e576d245892da980(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72826a8337ed93b7e576d245892da980(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72826a8337ed93b7e576d245892da980(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72826a8337ed93b7e576d245892da980(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72826a8337ed93b7e576d245892da980(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72826a8337ed93b7e576d245892da980(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72826a8337ed93b7e576d245892da980(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72826a8337ed93b7e576d245892da980(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72826a8337ed93b7e576d245892da980(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72826a8337ed93b7e576d245892da980(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_a99aa0d1919980cadc2fb18820335855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3813, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2e626db0c7c2e76d664154f33b1b8b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
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


class TestPrimitiveOp_c26b7d45767ca38ed487c66b43128704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5069563d8a8e8aaf71ce0ba65c42de9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
            paddle.to_tensor([0.43365156650543213, 0.14819391071796417, 0.36823219060897827, 0.11995603144168854, 0.29551056027412415, 0.16545988619327545, 0.23600615561008453, 0.33373603224754333, 0.3894231617450714, 0.2624466121196747, 0.3304726183414459, 0.04686542972922325, 0.17661075294017792, 0.16494564712047577, 0.3988954424858093, 0.017083849757909775, 0.04391861706972122, 0.3052978515625, 0.18097592890262604, 0.31674817204475403], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_b48f1312bec3eaec1aedb29742e1c298(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.43365156650543213, 0.14819391071796417, 0.36823219060897827, 0.11995603144168854, 0.29551056027412415, 0.16545988619327545, 0.23600615561008453, 0.33373603224754333, 0.3894231617450714, 0.2624466121196747, 0.3304726183414459, 0.04686542972922325, 0.17661075294017792, 0.16494564712047577, 0.3988954424858093, 0.017083849757909775, 0.04391861706972122, 0.3052978515625, 0.18097592890262604, 0.31674817204475403], dtype='float32').reshape([20]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_239c677de91b5cec3fb400c27c85d69b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14417004585266113], [0.1984117180109024], [0.015906473621726036], [0.1194428876042366]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.15914317965507507], [0.45126867294311523], [0.333740234375], [0.38055622577667236]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_57a32e0586b76022226ff7f286e6c593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2512052059173584], [0.3719286024570465], [0.06619598716497421], [0.10990522801876068]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4642404615879059], [0.36144939064979553], [0.43231305480003357], [0.17521435022354126]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_b12ba832bdbdbf209a74e7abb7969467(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14417004585266113], [0.38891172409057617], [0.17883332073688507], [0.1194428876042366]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.15914317965507507], [0.45126867294311523], [0.333740234375], [0.38055622577667236]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_bfea93073e58fef30665a97cec48fed1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2988428473472595], [0.4232253134250641], [0.06619598716497421], [0.38745006918907166]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4642404615879059], [0.014474418014287949], [0.04351218044757843], [0.07896348834037781]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_8d48bad250f579e3a7918be5180e0c60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17582319676876068], [0.1984117180109024], [0.015906473621726036], [0.22741685807704926]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.057106561958789825], [0.3722459077835083], [0.06349729001522064], [0.3261486887931824]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_06d66c5882591efeaec8c526534b157d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2512052059173584], [0.3719286024570465], [0.23222172260284424], [0.10990522801876068]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3617795407772064], [0.36144939064979553], [0.43231305480003357], [0.17521435022354126]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_87762811ff45374d28e294210181c978(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.010650492273271084], [-0.027310103178024292], [0.0060086315497756], [-0.07410187274217606]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_6bba2bb9da810627361918ad5d3f6272(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17582319676876068], [0.38891172409057617], [0.17883332073688507], [0.22741685807704926]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.057106561958789825], [0.3722459077835083], [0.06349729001522064], [0.3261486887931824]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_b0bdbdaa309a2c752028b985d2b1ee2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2988428473472595], [0.4232253134250641], [0.23222172260284424], [0.38745006918907166]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3617795407772064], [0.014474418014287949], [0.04351218044757843], [0.07896348834037781]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_fdbb34326deb77cbec153f11b8635157(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.007471632678061724], [0.006812167353928089], [0.021765010431408882], [-0.030457444489002228]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.010650492273271084], [-0.027310103178024292], [0.0060086315497756], [-0.07410187274217606]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_533191a436cf0a77cf29445d95117aff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [-0.0], [0.0], [-0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.4254571497440338], [5.0090179443359375], [0.7239316701889038], [-1.4329642057418823]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_78a96c00e7d75950e6e509e9740cb87f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dde7c4b3c800b733cbc29f44086791e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2100, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5639fc4c9213e74e3587c37053a91f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5639fc4c9213e74e3587c37053a91f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5639fc4c9213e74e3587c37053a91f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5639fc4c9213e74e3587c37053a91f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5639fc4c9213e74e3587c37053a91f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5639fc4c9213e74e3587c37053a91f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5639fc4c9213e74e3587c37053a91f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5639fc4c9213e74e3587c37053a91f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5639fc4c9213e74e3587c37053a91f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5639fc4c9213e74e3587c37053a91f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5639fc4c9213e74e3587c37053a91f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_dde7c4b3c800b733cbc29f44086791e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2100, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c26b7d45767ca38ed487c66b43128704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbfc37e66ed40be586cf03eae800a7e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_256a086b909e0fe2443a468ccc96885f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58b4e005dcab53bb7ba6fd2b2a4c72dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08064261078834534, 0.18323370814323425, 0.4429212808609009, 0.2096390575170517], [0.4880096912384033, 0.46780771017074585, 0.24352048337459564, 0.371028870344162], [0.18291239440441132, 0.1317790001630783, 0.43622735142707825, 0.33931058645248413], [0.18291239440441132, 0.1317790001630783, 0.43622735142707825, 0.33931058645248413], [0.3130415678024292, 0.2922552227973938, 0.29037177562713623, 0.16916565597057343]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.40394097566604614, 0.4475269615650177, 0.4100964665412903, 0.27925464510917664], [0.3794582188129425, 0.22883190214633942, 0.12280057370662689, 0.40007758140563965], [0.1433570683002472, 0.02906474657356739, 0.3163014054298401, 0.2556002736091614], [0.1433570683002472, 0.02906474657356739, 0.3163014054298401, 0.2556002736091614], [0.058678410947322845, 0.04930073022842407, 0.14398008584976196, 0.20175908505916595]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_d25078d804a8a7d72eccfbd7f7604302(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4231, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b603a4485cd0beb79a4bd076e947298a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b603a4485cd0beb79a4bd076e947298a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b603a4485cd0beb79a4bd076e947298a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b603a4485cd0beb79a4bd076e947298a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b603a4485cd0beb79a4bd076e947298a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b603a4485cd0beb79a4bd076e947298a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b603a4485cd0beb79a4bd076e947298a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b603a4485cd0beb79a4bd076e947298a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b603a4485cd0beb79a4bd076e947298a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b603a4485cd0beb79a4bd076e947298a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b603a4485cd0beb79a4bd076e947298a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_d25078d804a8a7d72eccfbd7f7604302(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4231, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d0ad1c7bfdff455b6b893986d047de5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3470168709754944, 0.25435441732406616, 0.3439519703388214, 0.06288041919469833], [0.4462828040122986, 0.2881370782852173, 0.2048557847738266, 0.392544150352478], [0.49107927083969116, 0.4175286591053009, 0.4714440107345581, 0.12655608355998993], [0.3470168709754944, 0.25435441732406616, 0.3439519703388214, 0.06288041919469833], [0.40785568952560425, 0.19114325940608978, 0.27876362204551697, 0.33430197834968567], [0.11329272389411926, 0.31143176555633545, 0.3399847149848938, 0.19061262905597687], [0.40785568952560425, 0.19114325940608978, 0.27876362204551697, 0.33430197834968567]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.14899523556232452, 0.42397019267082214, 0.19167467951774597, 0.03783583641052246], [0.21096032857894897, 0.2342609167098999, 0.35275283455848694, 0.32945728302001953], [0.4924883544445038, 0.46976780891418457, 0.38682612776756287, 0.195204496383667], [0.14899523556232452, 0.42397019267082214, 0.19167467951774597, 0.03783583641052246], [0.3735659420490265, 0.31074845790863037, 0.4845695197582245, 0.405693918466568], [0.19877874851226807, 0.23526626825332642, 0.04873793199658394, 0.3208487629890442], [0.3735659420490265, 0.31074845790863037, 0.4845695197582245, 0.405693918466568]], dtype='float32').reshape([7, 4]),
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