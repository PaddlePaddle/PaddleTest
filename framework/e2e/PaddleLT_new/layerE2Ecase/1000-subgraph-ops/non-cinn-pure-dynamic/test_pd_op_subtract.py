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


class TestPrimitiveOp_5b172b467f86fa3e49b324cede697124(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.16128204762935638]], [[0.3955680727958679]], [[0.3676556348800659]], [[0.07654835283756256]], [[0.21720686554908752]], [[0.32066860795021057]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.7415937781333923]], [[0.617002010345459]], [[0.6073431372642517]], [[0.5965418815612793]], [[0.6140335202217102]], [[0.7699944972991943]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_f196dd23b3f2284c7986971f009a0d7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.24854236841201782]], [[0.23063287138938904]], [[0.2910938858985901]], [[0.21472863852977753]], [[0.14672036468982697]], [[0.09137527644634247]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.8101144433021545]], [[0.8019925951957703]], [[0.6913491487503052]], [[0.6806997060775757]], [[0.5097703337669373]], [[0.7950479388237]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_f3f1c2ef42dbe3c62a22cd5c508a331c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.08875571936368942, 0.22509047389030457]], [[0.2513754069805145, 0.42446649074554443]], [[0.3137357234954834, 0.04039686918258667]], [[0.24674072861671448, 0.11493134498596191]], [[0.0873730331659317, 0.3213169276714325]], [[0.2712728679180145, 0.33490613102912903]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.08220051229000092, 0.06182434409856796]], [[0.4791288673877716, 0.12093376368284225]], [[0.12961630523204803, 0.45618799328804016]], [[0.11299704760313034, 0.4686511158943176]], [[0.001913706073537469, 0.498259037733078]], [[0.11343972384929657, 0.09046033769845963]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_94f1ace622d7b34d8f42d5d6f853c67f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.03313034027814865, 0.033602338284254074]], [[0.2645939290523529, 0.45329976081848145]], [[0.21010246872901917, 0.23237355053424835]], [[0.17494092881679535, 0.33330217003822327]], [[0.06913191825151443, 0.32164785265922546]], [[0.2734658718109131, 0.00511071365326643]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.08220051229000092, 0.06182434409856796]], [[0.4791288673877716, 0.12093376368284225]], [[0.12961630523204803, 0.45618799328804016]], [[0.11299704760313034, 0.4686511158943176]], [[0.001913706073537469, 0.498259037733078]], [[0.11343972384929657, 0.09046033769845963]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_dadafbba8c956b012cd294af629137ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.31379416584968567, 0.07449933141469955]], [[0.24354813992977142, 0.07788249850273132]], [[0.10838872194290161, 0.3051914870738983]], [[0.40201398730278015, 0.08278276771306992]], [[0.1669904887676239, 0.08771916478872299]], [[0.27185600996017456, 0.19672079384326935]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class TestPrimitiveOp_ca838ab7fae244326084a72d7c64d87c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
            paddle.to_tensor([0.39680296182632446, 0.4667986035346985, 0.19420655071735382, 0.0747121125459671, 0.10519072413444519, 0.340871125459671, 0.4899193346500397, 0.47285255789756775, 0.16072697937488556, 0.24498629570007324, 0.2651890218257904, 0.457339882850647, 0.4159230887889862, 0.09593425691127777, 0.09247507154941559, 0.21826285123825073], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_b1f8d3d7be2da81f833d2f5b1e0c07ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.39680296182632446, 0.4667986035346985, 0.19420655071735382, 0.0747121125459671, 0.10519072413444519, 0.340871125459671, 0.4899193346500397, 0.47285255789756775, 0.16072697937488556, 0.24498629570007324, 0.2651890218257904, 0.457339882850647, 0.4159230887889862, 0.09593425691127777, 0.09247507154941559, 0.21826285123825073], dtype='float32').reshape([16]),
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


class TestPrimitiveOp_44934f1e87b190eb17dc949bda0af603(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1762, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1762, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3dae754e16431811b8d966ec2c16e8b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3dae754e16431811b8d966ec2c16e8b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3dae754e16431811b8d966ec2c16e8b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3dae754e16431811b8d966ec2c16e8b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3dae754e16431811b8d966ec2c16e8b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3dae754e16431811b8d966ec2c16e8b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3dae754e16431811b8d966ec2c16e8b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3dae754e16431811b8d966ec2c16e8b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3dae754e16431811b8d966ec2c16e8b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3dae754e16431811b8d966ec2c16e8b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3dae754e16431811b8d966ec2c16e8b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_44934f1e87b190eb17dc949bda0af603(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1762, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1762, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3cf749f1c625e0eb9b6d4cff1effc9be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.29359763860702515, 0.35291045904159546, 0.20213532447814941, 0.04104747995734215], [0.3494870364665985, 0.10577014833688736, 0.2663863003253937, 0.14588382840156555], [0.4335102140903473, 0.36289742588996887, 0.3581336736679077, 0.18044042587280273], [0.1699855625629425, 0.005559444427490234, 0.2521253824234009, 0.14972518384456635], [0.4218791425228119, 0.03831281512975693, 0.08873697370290756, 0.21815598011016846]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.3551108241081238, 0.47227880358695984, 0.12412738054990768, 0.30854466557502747], [0.23553884029388428, 0.10051453113555908, 0.3624175786972046, 0.08863233029842377], [0.1522759348154068, 0.08564569801092148, 0.14454789459705353, 0.3739643692970276], [0.22875022888183594, 0.29198890924453735, 0.36239275336265564, 0.47895878553390503], [0.19388657808303833, 0.17426227033138275, 0.3238770663738251, 0.17398157715797424]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_6a52144d1e589b66c62325e1b947de26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.18183766305446625, 0.12805365025997162, 0.32253557443618774, 0.05932657793164253], [0.18703408539295197, 0.4966813325881958, 0.035800762474536896, 0.05232797935605049], [0.21098017692565918, 0.26502904295921326, 0.2809772193431854, 0.44455385208129883], [0.18703408539295197, 0.4966813325881958, 0.035800762474536896, 0.05232797935605049], [0.21098017692565918, 0.26502904295921326, 0.2809772193431854, 0.44455385208129883]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.4493182599544525, 0.26249080896377563, 0.30809953808784485, 0.4826532006263733], [0.30505335330963135, 0.32070192694664, 0.30816787481307983, 0.35593414306640625], [0.14694428443908691, 0.47813260555267334, 0.3056328296661377, 0.12107192724943161], [0.30505335330963135, 0.32070192694664, 0.30816787481307983, 0.35593414306640625], [0.14694428443908691, 0.47813260555267334, 0.3056328296661377, 0.12107192724943161]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_a11766378fc58ae2929ec31c896c1aa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1447327435016632], [0.08309690654277802], [0.038118865340948105], [0.015629051253199577], [0.41757383942604065], [0.012152628973126411], [0.4100273549556732], [0.30988749861717224], [0.04747486487030983]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3858015835285187], [0.3827413022518158], [0.40329664945602417], [0.3532903492450714], [0.46032440662384033], [0.3928758203983307], [0.4579012989997864], [0.31710684299468994], [0.4311077892780304]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_67b18f3816544c379a7562747cf8a9a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03195398673415184], [0.20543742179870605], [0.38529640436172485], [0.4782434105873108], [0.19919998943805695], [0.29198157787323], [0.0029337117448449135], [0.12914401292800903], [0.013538859784603119]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3669567406177521], [0.44098344445228577], [0.1773584485054016], [0.24322769045829773], [0.46047866344451904], [0.3803057074546814], [0.29980069398880005], [0.4812083840370178], [0.2672547399997711]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_ec9d49d9e902f0f8a0661b71ee69495e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.437514990568161], [0.08309690654277802], [0.46463990211486816], [0.015629051253199577], [0.41757383942604065], [0.012152628973126411], [0.48972272872924805], [0.30988749861717224], [0.16249637305736542]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.23177070915699005], [0.3827413022518158], [0.12799298763275146], [0.1969173699617386], [0.011041968129575253], [0.3928758203983307], [0.4579012989997864], [0.31710684299468994], [0.2874194085597992]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_dde3048e2332a24906f1c524c958c202(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03195398673415184], [0.4533325731754303], [0.38529640436172485], [0.4782434105873108], [0.19919998943805695], [0.4240538477897644], [0.22890320420265198], [0.41177159547805786], [0.28056761622428894]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3669567406177521], [0.44098344445228577], [0.14103317260742188], [0.07160888612270355], [0.12475907802581787], [0.3803057074546814], [0.29980069398880005], [0.4812083840370178], [0.2672547399997711]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_59438c0c23a45e21a58838db5240f9be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1447327435016632], [0.1211131140589714], [0.038118865340948105], [0.29171764850616455], [0.4933195412158966], [0.26447802782058716], [0.4100273549556732], [0.4631510376930237], [0.04747486487030983]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3858015835285187], [0.2665897011756897], [0.40329664945602417], [0.3532903492450714], [0.46032440662384033], [0.0526868961751461], [0.3947453498840332], [0.13757629692554474], [0.4311077892780304]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_41fbaab4d41458fb0a8c791d5128855b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0798618495464325], [0.20543742179870605], [0.4339882731437683], [0.4867236614227295], [0.4583660364151001], [0.29198157787323], [0.0029337117448449135], [0.12914401292800903], [0.013538859784603119]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.06200192868709564], [0.2388659417629242], [0.1773584485054016], [0.24322769045829773], [0.46047866344451904], [0.18422792851924896], [0.10528553277254105], [0.09368486702442169], [0.062429703772068024]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_5739c510cf07dedf43cb24d3d2c5c18b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.07323036342859268], [0.001162719214335084], [-0.01148504763841629], [-0.08871079236268997], [0.030192896723747253], [0.006165334954857826], [-0.003820200450718403], [0.012045891024172306], [0.017093053087592125]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_2f13994800f09d7b699d2141de071efc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.437514990568161], [0.1211131140589714], [0.46463990211486816], [0.29171764850616455], [0.4933195412158966], [0.26447802782058716], [0.48972272872924805], [0.4631510376930237], [0.16249637305736542]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.23177070915699005], [0.2665897011756897], [0.12799298763275146], [0.1969173699617386], [0.011041968129575253], [0.0526868961751461], [0.3947453498840332], [0.13757629692554474], [0.2874194085597992]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_78b5c641dc44f3792a7a999a7adba9a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0798618495464325], [0.4533325731754303], [0.4339882731437683], [0.4867236614227295], [0.4583660364151001], [0.4240538477897644], [0.22890320420265198], [0.41177159547805786], [0.28056761622428894]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.06200192868709564], [0.2388659417629242], [0.14103317260742188], [0.07160888612270355], [0.12475907802581787], [0.18422792851924896], [0.10528553277254105], [0.09368486702442169], [0.062429703772068024]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_9e0d599a8e71d81dfcf9ff7baf22997a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.00367457652464509], [-0.031199872493743896], [0.09862243384122849], [0.03935299441218376], [0.16089116036891937], [0.05079300329089165], [0.011740881949663162], [0.10356101393699646], [-0.02725045196712017]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.07323036342859268], [0.0011627193307504058], [-0.01148504763841629], [-0.08871079236268997], [0.030192896723747253], [0.006165334954857826], [-0.003820200450718403], [0.012045891024172306], [0.017093053087592125]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_0bd39b72a441155ff63473ff76d86129(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [0.0], [-0.0], [-0.0], [0.0], [0.0], [-0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[20.928924560546875], [1.037266731262207], [1.1164547204971313], [3.254232168197632], [0.8123396039009094], [0.8786183595657349], [1.3253759145736694], [0.8836831450462341], [1.6272575855255127]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_b683a7538b123f277ffef55f48129604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58e93cc613e2a3e0b00063e5cacf2846(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.45823708176612854]], [[0.489061176776886]], [[0.32814788818359375]], [[0.45978647470474243]], [[0.1830926537513733]], [[0.476858913898468]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6954591274261475]], [[0.6292228102684021]], [[0.5073301792144775]], [[0.5909470915794373]], [[0.7551741600036621]], [[0.6076744794845581]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_136452a34d7fba3d480f9e7ef416713b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.0194272231310606]], [[0.41314589977264404]], [[0.01989416405558586]], [[0.4264542758464813]], [[0.16482803225517273]], [[0.2548598349094391]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.7169739603996277]], [[0.6037969589233398]], [[0.6977351307868958]], [[0.5211019515991211]], [[0.5051576495170593]], [[0.6865183115005493]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_c31382c9e3e3a476da7fe5a44630f780(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_24e6eda947b85f9c8e09f8b955dc59ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5522, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5522, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba6397e443de4b90627952d1e7300a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba6397e443de4b90627952d1e7300a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba6397e443de4b90627952d1e7300a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba6397e443de4b90627952d1e7300a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba6397e443de4b90627952d1e7300a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba6397e443de4b90627952d1e7300a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba6397e443de4b90627952d1e7300a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba6397e443de4b90627952d1e7300a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba6397e443de4b90627952d1e7300a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba6397e443de4b90627952d1e7300a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba6397e443de4b90627952d1e7300a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_24e6eda947b85f9c8e09f8b955dc59ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([5522, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5522, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6af598aa278c6a4a184cd8eece2dd16c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.26998353004455566, 0.018573515117168427, 0.2389834225177765, 0.39205247163772583], [0.231656014919281, 0.30609434843063354, 0.022763900458812714, 0.41933590173721313], [0.41856929659843445, 0.4133129417896271, 0.3632120192050934, 0.47915127873420715], [0.231656014919281, 0.30609434843063354, 0.022763900458812714, 0.41933590173721313], [0.41856929659843445, 0.4133129417896271, 0.3632120192050934, 0.47915127873420715], [0.2596732974052429, 0.3161533772945404, 0.41002994775772095, 0.328370064496994], [0.2596732974052429, 0.3161533772945404, 0.41002994775772095, 0.328370064496994]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.2005547434091568, 0.41836050152778625, 0.2798846364021301, 0.27603182196617126], [0.1754767894744873, 0.09332948923110962, 0.1401137411594391, 0.3314807713031769], [0.25368696451187134, 0.4652036130428314, 0.34856608510017395, 0.40978100895881653], [0.1754767894744873, 0.09332948923110962, 0.1401137411594391, 0.3314807713031769], [0.25368696451187134, 0.4652036130428314, 0.34856608510017395, 0.40978100895881653], [0.4619261920452118, 0.06540755927562714, 0.18247419595718384, 0.38211148977279663], [0.4619261920452118, 0.06540755927562714, 0.18247419595718384, 0.38211148977279663]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_302a838be5d796c8be94c12728edf327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4128625690937042, 0.3836251497268677, 0.3297223746776581, 0.1599111407995224, 0.4577946364879608, 0.3183137774467468], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3605007231235504, 0.3333803117275238, 0.23454944789409637, 0.26296818256378174, 0.3854544162750244, 0.2304389327764511], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_1af96ef141561192ee632e549693b9f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.023622216656804085, 0.2635997235774994, 0.20144450664520264, 0.25364255905151367, 0.04704642668366432, 0.07614580541849136], dtype='float32').reshape([6]),
            paddle.to_tensor([0.42629650235176086, 0.18972466886043549, 0.20663975179195404, 0.15081170201301575, 0.4174017608165741, 0.3629100024700165], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_d00e073578a1186714e1dd3b5b397260(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0922485888004303, 0.42701223492622375, 0.3167136311531067, 0.34751439094543457, 0.4444928765296936, 0.42101287841796875], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4335317313671112, 0.11488522589206696, 0.34194254875183105, 0.16626453399658203, 0.2805379629135132, 0.4297538995742798], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_c39535f39e7f7095edf28650ecfea189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1620912402868271, 0.20841006934642792, 0.06403285264968872, 0.4416305720806122, 0.15589286386966705, 0.45514115691185], dtype='float32').reshape([6]),
            paddle.to_tensor([0.013391555286943913, 0.3437585234642029, 0.4915377199649811, 0.19970282912254333, 0.21376128494739532, 0.1659878045320511], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_6c430c0b2041377544bdcae6436fe7ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0922485888004303, 0.3836251497268677, 0.3167136311531067, 0.26296818256378174, 0.4444928765296936, 0.3183137774467468], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4335317313671112, 0.3333803117275238, 0.34194254875183105, 0.26296818256378174, 0.3854544162750244, 0.4297538995742798], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_e8203beda5d7d9e26d3d55ce600912e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1620912402868271, 0.20841006934642792, 0.06403285264968872, 0.25364255905151367, 0.15589286386966705, 0.3629100024700165], dtype='float32').reshape([6]),
            paddle.to_tensor([0.42629650235176086, 0.3437585234642029, 0.4915377199649811, 0.19970282912254333, 0.4174017608165741, 0.3629100024700165], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_7b15da1d6317da85d23ac1df691877d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4128625690937042, 0.3836251497268677, 0.3297223746776581, 0.26296818256378174, 0.4577946364879608, 0.3183137774467468], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3605007231235504, 0.3333803117275238, 0.23454944789409637, 0.26296818256378174, 0.3854544162750244, 0.2304389327764511], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_e0f9c935f63508177434447e83b54b70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.42629650235176086, 0.2635997235774994, 0.20663975179195404, 0.25364255905151367, 0.4174017608165741, 0.3629100024700165], dtype='float32').reshape([6]),
            paddle.to_tensor([0.42629650235176086, 0.18972466886043549, 0.20663975179195404, 0.15081170201301575, 0.4174017608165741, 0.3629100024700165], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_d69d62ae593d74808201f10a3e9a528d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.05074869468808174, -0.03853406757116318, 0.010785484686493874, 0.04384936764836311, -0.009487812407314777, -0.0025274953804910183], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, -0.0, 0.0, 0.0, -0.0, -0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_0a3719b0ac6b092824cb1a9540db6479(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3866816461086273, 0.35850274562835693, 0.28213590383529663, 0.21143966913223267, 0.4216245412826538, 0.27437636256217957], dtype='float32').reshape([6]),
            paddle.to_tensor([0.26289016008377075, 0.27094873785972595, 0.32932808995246887, 0.2568894624710083, 0.3625154197216034, 0.42538338899612427], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_72d131793b46c8fdf48cea66a6c58f16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2249593585729599, 0.22666218876838684, 0.20404213666915894, 0.2022271305322647, 0.23222409188747406, 0.21952790021896362], dtype='float32').reshape([6]),
            paddle.to_tensor([0.08774139732122421, 0.276084303855896, 0.2777853012084961, 0.32066670060157776, 0.1848270744085312, 0.31056448817253113], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_9c206641595b3ac3ae00729ee851e48f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4128625690937042, 0.42701223492622375, 0.3297223746776581, 0.34751439094543457, 0.4577946364879608, 0.42101287841796875], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3605007231235504, 0.11488522589206696, 0.23454944789409637, 0.16626453399658203, 0.2805379629135132, 0.2304389327764511], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_2f37f5a4d6f265db67763183034a5730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.42629650235176086, 0.2635997235774994, 0.20663975179195404, 0.4416305720806122, 0.4174017608165741, 0.45514115691185], dtype='float32').reshape([6]),
            paddle.to_tensor([0.013391555286943913, 0.18972466886043549, 0.20663975179195404, 0.15081170201301575, 0.21376128494739532, 0.1659878045320511], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_cd2281f2880af1d442d9cc5ffbc2686f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.1598912477493286, -1.1616365909576416, 0.058945972472429276, 0.642982542514801, -1.2314929962158203, -0.030220504850149155], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.12930965423583984, 0.5972673892974854, -1.5162630081176758, -0.7864967584609985, -0.19289782643318176, -0.29735076427459717], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_db6e5554c6d5c349f099f35a453e750c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1760, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1760, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54fc5b3c52a68a19a56df0b1f1b06832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54fc5b3c52a68a19a56df0b1f1b06832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54fc5b3c52a68a19a56df0b1f1b06832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54fc5b3c52a68a19a56df0b1f1b06832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54fc5b3c52a68a19a56df0b1f1b06832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54fc5b3c52a68a19a56df0b1f1b06832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54fc5b3c52a68a19a56df0b1f1b06832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54fc5b3c52a68a19a56df0b1f1b06832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54fc5b3c52a68a19a56df0b1f1b06832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54fc5b3c52a68a19a56df0b1f1b06832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54fc5b3c52a68a19a56df0b1f1b06832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_db6e5554c6d5c349f099f35a453e750c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1760, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1760, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c46db15154ef35bed66e9dd39e5923d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e13a6397d10d196f2b4b58d8f550f75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
            paddle.to_tensor([0.005897914059460163, 0.3780747950077057, 0.3955202102661133, 0.08381230384111404, 0.1808137446641922, 0.17841923236846924, 0.4738955795764923, 0.2507706582546234, 0.3080895245075226, 0.38261720538139343, 0.3476063311100006, 0.31858983635902405, 0.22009709477424622, 0.09035298973321915, 0.23606078326702118, 0.0701029971241951, 0.26301389932632446, 0.37287917733192444, 0.009290380403399467, 0.41418489813804626, 0.0361352413892746, 0.08704117685556412, 0.01657458208501339, 0.14088433980941772], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_e233310855dd5e12d76919865c9d6527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.005897914059460163, 0.3780747950077057, 0.3955202102661133, 0.08381230384111404, 0.1808137446641922, 0.17841923236846924, 0.4738955795764923, 0.2507706582546234, 0.3080895245075226, 0.38261720538139343, 0.3476063311100006, 0.31858983635902405, 0.22009709477424622, 0.09035298973321915, 0.23606078326702118, 0.0701029971241951, 0.26301389932632446, 0.37287917733192444, 0.009290380403399467, 0.41418489813804626, 0.0361352413892746, 0.08704117685556412, 0.01657458208501339, 0.14088433980941772], dtype='float32').reshape([24]),
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


class TestPrimitiveOp_f371d373c64a834d74fc6b9fbb125577(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1522, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1522, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06e814ef4e799d133b32422c58759007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06e814ef4e799d133b32422c58759007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06e814ef4e799d133b32422c58759007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06e814ef4e799d133b32422c58759007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06e814ef4e799d133b32422c58759007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06e814ef4e799d133b32422c58759007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06e814ef4e799d133b32422c58759007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06e814ef4e799d133b32422c58759007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06e814ef4e799d133b32422c58759007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06e814ef4e799d133b32422c58759007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06e814ef4e799d133b32422c58759007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f371d373c64a834d74fc6b9fbb125577(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1522, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1522, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_ea1c624e77b0542b115825b425ac8c46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
            paddle.to_tensor([0.09833092987537384, 0.18489766120910645, 0.14477917551994324, 0.3833272457122803], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_86c3916b072af06d872f77a837036138(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09833092987537384, 0.18489766120910645, 0.14477917551994324, 0.3833272457122803], dtype='float32').reshape([4]),
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


class TestPrimitiveOp_fe86a01b31aa7692bac13b1cf359a59e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.46209558844566345, 0.2259654402732849, 0.35871621966362, 0.10184191167354584], [0.15963271260261536, 0.1272817850112915, 0.38483911752700806, 0.4967978298664093], [0.3575408458709717, 0.13848216831684113, 0.31636548042297363, 0.443376749753952], [0.06935324519872665, 0.45508792996406555, 0.11233685910701752, 0.0742255300283432], [0.06935324519872665, 0.45508792996406555, 0.11233685910701752, 0.0742255300283432], [0.3575408458709717, 0.13848216831684113, 0.31636548042297363, 0.443376749753952]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.35227733850479126, 0.04416180029511452, 0.35333192348480225, 0.27619144320487976], [0.035690583288669586, 0.47501805424690247, 0.2805340588092804, 0.2660118639469147], [0.09601441770792007, 0.004303815774619579, 0.063333660364151, 0.3509784936904907], [0.48407283425331116, 0.16244995594024658, 0.2126832902431488, 0.4727063775062561], [0.48407283425331116, 0.16244995594024658, 0.2126832902431488, 0.4727063775062561], [0.09601441770792007, 0.004303815774619579, 0.063333660364151, 0.3509784936904907]], dtype='float32').reshape([6, 4]),
        ]


class TestPrimitiveOp_2c65d2ab0fba72df6ad31bdd06d0551c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19057437777519226, 0.24791334569454193, 0.49417755007743835, 0.26402318477630615], [0.13720861077308655, 0.3495766520500183, 0.0821411982178688, 0.4190191328525543], [0.040898095816373825, 0.1991966813802719, 0.19369058310985565, 0.024699218571186066], [0.2716957926750183, 0.4592864513397217, 0.24113377928733826, 0.49265822768211365], [0.19057437777519226, 0.24791334569454193, 0.49417755007743835, 0.26402318477630615]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.06166596710681915, 0.0021998989395797253, 0.22934894263744354, 0.07631295919418335], [0.09348897635936737, 0.38592201471328735, 0.029284100979566574, 0.3620889186859131], [0.3878341019153595, 0.38617339730262756, 0.2761750817298889, 0.2624339163303375], [0.08592834323644638, 0.39043545722961426, 0.3863198459148407, 0.4585285484790802], [0.06166596710681915, 0.0021998989395797253, 0.22934894263744354, 0.07631295919418335]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_dc38fbca4d28435e70b9ea4f76b0680d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a1704c3e1af18321dadfbc06f47cdd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06771773844957352]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.1288921982049942]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_ec8a156c880e77a9c90249a3121c247b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15218256413936615]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3986872434616089]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_08e486c26a6e8b32d60215509ea5b7cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06771773844957352]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0983637347817421]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_2883febf9eddf7351814e6e55b05842b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15742851793766022]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.2091304212808609]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_310b2ee673ab3fc5be7839610308b6c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11078812927007675]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.1288921982049942]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_ec8a156c880e77a9c90249a3121c247b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15218256413936615]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3986872434616089]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_37cb6af2d4dbf5f55d0f9bb0ca23cae2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.006047193892300129]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_4b8df1abebf3954a96b746e457ca5751(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11078812927007675]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0983637347817421]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_2883febf9eddf7351814e6e55b05842b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15742851793766022]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.2091304212808609]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_364f64eb5ae3a9b1254a0552ee480e34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0006423647282645106]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.006047193892300129]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_57dbb32539beb0aa34ad6b32389f0a19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[10.413957595825195]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_65d37b57cf468c14688f835b3b59340f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22387434542179108], [0.044702861458063126], [0.20448708534240723], [0.24163779616355896], [0.03669232502579689], [0.17557862401008606]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.22831673920154572], [0.33949005603790283], [0.2869625687599182], [0.4246893525123596], [0.16642236709594727], [0.31202274560928345]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_c392cfeded04d155d15414a0ed1ecdb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20023022592067719], [0.2667973041534424], [0.011914309114217758], [0.16640233993530273], [0.04521951824426651], [0.09507034718990326]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.1350128948688507], [0.47312918305397034], [0.3752787411212921], [0.46865251660346985], [0.48315006494522095], [0.45246943831443787]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_f96a682af4c4667f4310b67591770597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.24626027047634125], [0.14295130968093872], [0.4793103039264679], [0.25223153829574585], [0.03669232502579689], [0.17557862401008606]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.22831673920154572], [0.33949005603790283], [0.2869625687599182], [0.4246893525123596], [0.002240711124613881], [0.14619268476963043]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_c8a06a6b701eb98581a928837ad254d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20023022592067719], [0.2667973041534424], [0.011914309114217758], [0.18980513513088226], [0.43041738867759705], [0.09507034718990326]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.1350128948688507], [0.47312918305397034], [0.32580894231796265], [0.46865251660346985], [0.4675268828868866], [0.08211576193571091]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_f89a42652c20f258042da9ce13a92b78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22387434542179108], [0.044702861458063126], [0.20448708534240723], [0.24163779616355896], [0.2939894199371338], [0.43704453110694885]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.015245765447616577], [0.23714831471443176], [0.2166336476802826], [0.3967365324497223], [0.16642236709594727], [0.31202274560928345]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_d722460b5f41bd410b03be790c00dc19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4093327224254608], [0.3914657533168793], [0.4158235192298889], [0.16640233993530273], [0.04521951824426651], [0.3970659077167511]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.1296968311071396], [0.11471404135227203], [0.3752787411212921], [0.31934216618537903], [0.48315006494522095], [0.45246943831443787]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_aa21e77302a6b4c8fb3534777a8b765d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05951027199625969], [-0.01270739734172821], [-0.06086939945816994], [0.07181018590927124], [-0.05714399367570877], [-0.006545965559780598]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_ece901b335bc1fb2f0e5da8e18fb481d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.24626027047634125], [0.14295130968093872], [0.4793103039264679], [0.25223153829574585], [0.2939894199371338], [0.43704453110694885]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.015245765447616577], [0.23714831471443176], [0.2166336476802826], [0.3967365324497223], [0.002240711124613881], [0.14619268476963043]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_7e9fef6803333facdd32070430d52d07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4093327224254608], [0.3914657533168793], [0.4158235192298889], [0.18980513513088226], [0.43041738867759705], [0.3970659077167511]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.1296968311071396], [0.11471404135227203], [0.32580894231796265], [0.31934216618537903], [0.4675268828868866], [0.08211576193571091]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_129f0278bcf26a09066325e703e08949(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06459995359182358], [-0.02606918103992939], [0.02364472858607769], [0.018718747422099113], [-0.010826646350324154], [0.0916038230061531]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.05951027199625969], [-0.01270739734172821], [-0.06086939945816994], [0.07181018590927124], [-0.05714399367570877], [-0.006545965559780598]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_ccf7ae3b636732fbf615d4b7fff9a2fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [-0.0], [-0.0], [0.0], [-0.0], [-0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.07878769934177399], [0.512550950050354], [3.5743327140808105], [-2.836270809173584], [-4.278088092803955], [1.0714595317840576]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_3718545c83119479494b03e026c2c1b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3379237949848175, 0.42591768503189087, 0.46721577644348145, 0.02651386708021164], [0.30243924260139465, 0.016519716009497643, 0.31733644008636475, 0.4360567629337311], [0.40742602944374084, 0.3983646631240845, 0.34221866726875305, 0.11806876212358475], [0.33260416984558105, 0.3714119791984558, 0.12883210182189941, 0.4057803750038147]], dtype='float32').reshape([4, 4]),
            paddle.to_tensor([[0.3076942265033722, 0.2722072899341583, 0.2481198012828827, 0.030001480132341385], [0.204896941781044, 0.1539415717124939, 0.0744905173778534, 0.37445273995399475], [0.47109317779541016, 0.29098615050315857, 0.035117559134960175, 0.08401335030794144], [0.42889466881752014, 0.37159234285354614, 0.2610809803009033, 0.004416251555085182]], dtype='float32').reshape([4, 4]),
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


class TestPrimitiveOp_4a186b227e9196329a919b6088ebea5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2074, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2074, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f19c05243d3b4b9ee46c1421d31481d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f19c05243d3b4b9ee46c1421d31481d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f19c05243d3b4b9ee46c1421d31481d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f19c05243d3b4b9ee46c1421d31481d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f19c05243d3b4b9ee46c1421d31481d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f19c05243d3b4b9ee46c1421d31481d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f19c05243d3b4b9ee46c1421d31481d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f19c05243d3b4b9ee46c1421d31481d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f19c05243d3b4b9ee46c1421d31481d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f19c05243d3b4b9ee46c1421d31481d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f19c05243d3b4b9ee46c1421d31481d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_4a186b227e9196329a919b6088ebea5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2074, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2074, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f41dfe641d17b6029caa2576f38c957(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.00629150727763772, 0.45306235551834106, 0.2936408519744873, 0.09138627350330353], [0.00629150727763772, 0.45306235551834106, 0.2936408519744873, 0.09138627350330353], [0.18775774538516998, 0.48869022727012634, 0.11088680475950241, 0.35055941343307495], [0.2914629578590393, 0.2643881142139435, 0.1897423416376114, 0.0198875293135643], [0.2820081412792206, 0.41779136657714844, 0.3895058333873749, 0.40036800503730774], [0.33072537183761597, 0.3509145975112915, 0.3924933075904846, 0.3774349093437195], [0.07152549922466278, 0.051878683269023895, 0.16920723021030426, 0.4353562891483307]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.18539854884147644, 0.10143377631902695, 0.4127208888530731, 0.33707886934280396], [0.18539854884147644, 0.10143377631902695, 0.4127208888530731, 0.33707886934280396], [0.16525773704051971, 0.1371699869632721, 0.26213452219963074, 0.2564891278743744], [0.12244774401187897, 0.025359969586133957, 0.022115860134363174, 0.30843639373779297], [0.4341961741447449, 0.1457686722278595, 0.4619542360305786, 0.2920302450656891], [0.1915431022644043, 0.4143870174884796, 0.11761119216680527, 0.07290829718112946], [0.09737962484359741, 0.17279000580310822, 0.41893312335014343, 0.47181716561317444]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_50051027d63dd25dcf013245d67b2d88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4734, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4734, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2abb2476eb64f6f4acdc4ff77f448244(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2abb2476eb64f6f4acdc4ff77f448244(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2abb2476eb64f6f4acdc4ff77f448244(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2abb2476eb64f6f4acdc4ff77f448244(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2abb2476eb64f6f4acdc4ff77f448244(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2abb2476eb64f6f4acdc4ff77f448244(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2abb2476eb64f6f4acdc4ff77f448244(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2abb2476eb64f6f4acdc4ff77f448244(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2abb2476eb64f6f4acdc4ff77f448244(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2abb2476eb64f6f4acdc4ff77f448244(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2abb2476eb64f6f4acdc4ff77f448244(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_50051027d63dd25dcf013245d67b2d88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4734, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4734, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_adb62690e661f370026c921fbfb496c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1074, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1074, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6748f345b0a4667a55883bfc9c3e52c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6748f345b0a4667a55883bfc9c3e52c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6748f345b0a4667a55883bfc9c3e52c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6748f345b0a4667a55883bfc9c3e52c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6748f345b0a4667a55883bfc9c3e52c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6748f345b0a4667a55883bfc9c3e52c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6748f345b0a4667a55883bfc9c3e52c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6748f345b0a4667a55883bfc9c3e52c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6748f345b0a4667a55883bfc9c3e52c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6748f345b0a4667a55883bfc9c3e52c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6748f345b0a4667a55883bfc9c3e52c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_adb62690e661f370026c921fbfb496c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([1074, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1074, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c9460cd58abff38d75364249aa3e12c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_733968b8fab0c85b6b2c58a6289a8906(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08295142650604248, 0.16166016459465027, 0.1410202980041504, 0.23604877293109894], [0.33173003792762756, 0.45231127738952637, 0.23738862574100494, 0.17031387984752655], [0.33173003792762756, 0.45231127738952637, 0.23738862574100494, 0.17031387984752655], [0.15963514149188995, 0.37651148438453674, 0.45985519886016846, 0.3500175178050995], [0.24929045140743256, 0.1447683721780777, 0.30041152238845825, 0.38289979100227356], [0.426910400390625, 0.39620187878608704, 0.400075763463974, 0.19106699526309967]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.2616228461265564, 0.20029567182064056, 0.2504124045372009, 0.012093265540897846], [0.3155108392238617, 0.15163302421569824, 0.27079296112060547, 0.23608294129371643], [0.3155108392238617, 0.15163302421569824, 0.27079296112060547, 0.23608294129371643], [0.44253185391426086, 0.48768138885498047, 0.06034601479768753, 0.24875320494174957], [0.47304922342300415, 0.1725447028875351, 0.3174428343772888, 0.3686710000038147], [0.2449370175600052, 0.08249155431985855, 0.12680037319660187, 0.46628960967063904]], dtype='float32').reshape([6, 4]),
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


class TestPrimitiveOp_948e7bcfdb93e41662e65bbf0b6d57ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e550e076a243ef110a84edc2744e886e
    def get_inputs(self):
        return [
            paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.6935383677482605, 0.10847660899162292, 1.2856578826904297, 15.21419906616211], [8.063020706176758, 0.5627660155296326, 0.3087352216243744, 1.1819560527801514]], dtype='float32').reshape([2, 4]),
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


class TestPrimitiveOp_024637e1454119aa7273cc520fc068b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e550e076a243ef110a84edc2744e886e
    def get_inputs(self):
        return [
            paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.5057370662689209, 0.812700092792511, 0.785373866558075, 4.298986434936523], [24.259803771972656, 1.3584874868392944, 0.42290088534355164, 0.18747907876968384]], dtype='float32').reshape([2, 4]),
        ]


class TestPrimitiveOp_8000866aa207e0d049685b7bb0a0a651(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10796797275543213], [0.07695849984884262], [0.11816670745611191], [0.017290905117988586], [0.11618880927562714]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.17697207629680634], [0.29903921484947205], [0.13782429695129395], [0.46026167273521423], [0.38709262013435364]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_32484bcb5f2a3a7fac20c863677a6dae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15855374932289124], [0.13624975085258484], [0.05144762992858887], [0.2953103184700012], [0.01684260368347168]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.39805009961128235], [0.4595024883747101], [0.12538257241249084], [0.4333687126636505], [0.47472846508026123]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_6221550a90e970219b9b2192c2fc5fca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10796797275543213], [0.4512024223804474], [0.11816670745611191], [0.465546578168869], [0.21543511748313904]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.17697207629680634], [0.04053914546966553], [0.00564520712941885], [0.05893722176551819], [0.38709262013435364]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_54835d2c96a5d152a35a46f5c3a7d72a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15855374932289124], [0.13624975085258484], [0.12530755996704102], [0.2953103184700012], [0.01684260368347168]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.39805009961128235], [0.3606886863708496], [0.03755515068769455], [0.2986750900745392], [0.32391923666000366]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_cd128b1211b92e41565b339cf8e8dd23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.42674073576927185], [0.07695849984884262], [0.3584158718585968], [0.017290905117988586], [0.11618880927562714]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0247951690107584], [0.29903921484947205], [0.13782429695129395], [0.46026167273521423], [0.1342550814151764]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_8f76570e384f016905dad8366308ca6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.28606072068214417], [0.3674119710922241], [0.05144762992858887], [0.40608474612236023], [0.32032695412635803]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.08429738134145737], [0.4595024883747101], [0.12538257241249084], [0.4333687126636505], [0.47472846508026123]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_46bc0284679b64f1c45d7a201dd7f041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09762410819530487], [-0.07171730697154999], [-0.006435392424464226], [0.010717852041125298], [0.05550146847963333]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_4a07bde3cbf5f21ad464123e4ef3ff7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.42674073576927185], [0.4512024223804474], [0.3584158718585968], [0.465546578168869], [0.21543511748313904]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0247951690107584], [0.04053914546966553], [0.00564520712941885], [0.05893722176551819], [0.1342550814151764]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_4ba522031c8d64b25c21d4c6db1a1981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.28606072068214417], [0.3674119710922241], [0.12530755996704102], [0.40608474612236023], [0.32032695412635803]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.08429738134145737], [0.3606886863708496], [0.03755515068769455], [0.2986750900745392], [0.32391923666000366]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_92ee9c76b3b84f985eb40aa1f7d21887(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08109787851572037], [0.002761006122455001], [0.03095647506415844], [0.04367377236485481], [-0.00029162154532969]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.09762410819530487], [-0.07171730697154999], [-0.006435392424464226], [0.010717852041125298], [0.05550146847963333]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_79b24d5ffe9de21a0bc41aea0bd229ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [-0.0], [-0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.20378127694129944], [26.97506332397461], [1.2078851461410522], [0.7545929551124573], [191.3201904296875]], dtype='float32').reshape([5, 1]),
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


class TestPrimitiveOp_1957ab5d4bffdcedcac28aa9e765c43d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2332, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2332, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6ca4d0ee2766f9777c2daab3f748de0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6ca4d0ee2766f9777c2daab3f748de0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6ca4d0ee2766f9777c2daab3f748de0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6ca4d0ee2766f9777c2daab3f748de0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6ca4d0ee2766f9777c2daab3f748de0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6ca4d0ee2766f9777c2daab3f748de0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6ca4d0ee2766f9777c2daab3f748de0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6ca4d0ee2766f9777c2daab3f748de0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6ca4d0ee2766f9777c2daab3f748de0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6ca4d0ee2766f9777c2daab3f748de0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6ca4d0ee2766f9777c2daab3f748de0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_1957ab5d4bffdcedcac28aa9e765c43d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2332, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2332, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a8bc92a939a81a19d37236153a0b5743(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3051, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3051, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7eeaa47443ee0ec420ebe7180445db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7eeaa47443ee0ec420ebe7180445db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7eeaa47443ee0ec420ebe7180445db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7eeaa47443ee0ec420ebe7180445db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7eeaa47443ee0ec420ebe7180445db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7eeaa47443ee0ec420ebe7180445db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7eeaa47443ee0ec420ebe7180445db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7eeaa47443ee0ec420ebe7180445db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7eeaa47443ee0ec420ebe7180445db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7eeaa47443ee0ec420ebe7180445db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7eeaa47443ee0ec420ebe7180445db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_a8bc92a939a81a19d37236153a0b5743(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3051, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3051, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a883badebd146548c7dfeb3c191368ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3870, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3870, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2172d6de8d15c933fe895498ae6c5387(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2172d6de8d15c933fe895498ae6c5387(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2172d6de8d15c933fe895498ae6c5387(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2172d6de8d15c933fe895498ae6c5387(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2172d6de8d15c933fe895498ae6c5387(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2172d6de8d15c933fe895498ae6c5387(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2172d6de8d15c933fe895498ae6c5387(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2172d6de8d15c933fe895498ae6c5387(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2172d6de8d15c933fe895498ae6c5387(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2172d6de8d15c933fe895498ae6c5387(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2172d6de8d15c933fe895498ae6c5387(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_a883badebd146548c7dfeb3c191368ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([3870, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3870, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_7d9bdeb73859631d9ae62edc1d8c553c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
            paddle.to_tensor([0.4408528804779053, 0.4109119772911072, 0.16994760930538177, 0.2217721790075302, 0.22215040028095245, 0.35895657539367676, 0.052866388112306595, 0.4876033365726471, 0.08720830827951431, 0.2539161145687103, 0.3830873370170593, 0.07409476488828659, 0.3757791519165039, 0.418334037065506, 0.17667099833488464, 0.16370463371276855, 0.280592679977417, 0.0035582492128014565, 0.2064770609140396, 0.3651828169822693], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_c82c61ee1603ebe3ff1a130dcd03983e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4408528804779053, 0.4109119772911072, 0.16994760930538177, 0.2217721790075302, 0.22215040028095245, 0.35895657539367676, 0.052866388112306595, 0.4876033365726471, 0.08720830827951431, 0.2539161145687103, 0.3830873370170593, 0.07409476488828659, 0.3757791519165039, 0.418334037065506, 0.17667099833488464, 0.16370463371276855, 0.280592679977417, 0.0035582492128014565, 0.2064770609140396, 0.3651828169822693], dtype='float32').reshape([20]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_b294e2e37025d6e05c612f99a7d36fe7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.006803394760936499], [0.1294925957918167], [0.19429393112659454], [0.0008755988092161715]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4567643404006958], [0.4713914096355438], [0.25339508056640625], [0.2674586772918701]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_254d5b9b2e6bc3524187f77c7905d74b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.21126195788383484], [0.052910272032022476], [0.1812242865562439], [0.038458243012428284]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2055460810661316], [0.26892995834350586], [0.4244452118873596], [0.09755375981330872]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_4ad102c45e644cfc1f175aee731653dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.006803394760936499], [0.40538290143013], [0.19567006826400757], [0.4959845542907715]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4567643404006958], [0.008849719539284706], [0.003253927920013666], [0.2674586772918701]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_fedec1de0b8d034a12ccd12cf742edbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.21126195788383484], [0.052910272032022476], [0.37054139375686646], [0.038458243012428284]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.11311373114585876], [0.26892995834350586], [0.4244452118873596], [0.09755375981330872]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_8fafb0f7e0a01b9ec43912cc93cd86f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.18735556304454803], [0.1294925957918167], [0.19429393112659454], [0.0008755988092161715]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3858823776245117], [0.4713914096355438], [0.25339508056640625], [0.21384595334529877]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_edbe6a8f62558400c7083117924bb764(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3032015860080719], [0.3202653229236603], [0.1812242865562439], [0.20975439250469208]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2055460810661316], [0.16424299776554108], [0.13470056653022766], [0.07356300950050354]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_f8b203c5415c34cd82fb125f3c2c43ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.06355010718107224], [-0.13900282979011536], [-0.013121570460498333], [-0.04250958189368248]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_12231d3f301be4c2f57ea3f57652301a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.18735556304454803], [0.40538290143013], [0.19567006826400757], [0.4959845542907715]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3858823776245117], [0.008849719539284706], [0.003253927920013666], [0.21384595334529877]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_06ebd93e0dd7f88cb3173309ba703332(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3032015860080719], [0.3202653229236603], [0.37054139375686646], [0.20975439250469208]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.11311373114585876], [0.16424299776554108], [0.13470056653022766], [0.07356300950050354]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_5d56b3dfaf73eb3d6388273b3753ed35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.037737537175416946], [0.061868030577898026], [0.04537958279252052], [0.03842484578490257]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.06355010718107224], [-0.13900282979011536], [-0.013121570460498333], [-0.04250958189368248]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_e93db19638b5f2c4d4f5f362141ebd7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [-0.0], [-0.0], [-0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.6840025186538696], [3.2467634677886963], [1.2891514301300049], [2.10630464553833]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_78a96c00e7d75950e6e509e9740cb87f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87006deab02d67727c3b070619eca920(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2111, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2111, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4add6eeb5aef63f504b32aaa736b46f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4add6eeb5aef63f504b32aaa736b46f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4add6eeb5aef63f504b32aaa736b46f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4add6eeb5aef63f504b32aaa736b46f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4add6eeb5aef63f504b32aaa736b46f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4add6eeb5aef63f504b32aaa736b46f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4add6eeb5aef63f504b32aaa736b46f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4add6eeb5aef63f504b32aaa736b46f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4add6eeb5aef63f504b32aaa736b46f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4add6eeb5aef63f504b32aaa736b46f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4add6eeb5aef63f504b32aaa736b46f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_87006deab02d67727c3b070619eca920(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([2111, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2111, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_33568bff10f21f9b26c2b96200475bf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3223201036453247, 0.1260082870721817, 0.3679535984992981, 0.2656900882720947], [0.1327272355556488, 0.024533171206712723, 0.2202073186635971, 0.31246405839920044], [0.42826804518699646, 0.041805852204561234, 0.2539118528366089, 0.014961760491132736], [0.42826804518699646, 0.041805852204561234, 0.2539118528366089, 0.014961760491132736], [0.26179933547973633, 0.07822809368371964, 0.28047803044319153, 0.47440141439437866]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.4570547640323639, 0.2488672435283661, 0.2623122036457062, 0.017825959250330925], [0.48248839378356934, 0.34005239605903625, 0.48969602584838867, 0.19127966463565826], [0.21359114348888397, 0.4423896074295044, 0.3889133632183075, 0.14987467229366302], [0.21359114348888397, 0.4423896074295044, 0.3889133632183075, 0.14987467229366302], [0.2678714096546173, 0.2697119414806366, 0.3073708117008209, 0.1718270480632782]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_4cb9ca95528e2061ba7ada35a9e930ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4141, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4141, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecf220a72955018595fc945e9410ae59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecf220a72955018595fc945e9410ae59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecf220a72955018595fc945e9410ae59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecf220a72955018595fc945e9410ae59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecf220a72955018595fc945e9410ae59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecf220a72955018595fc945e9410ae59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecf220a72955018595fc945e9410ae59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecf220a72955018595fc945e9410ae59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecf220a72955018595fc945e9410ae59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecf220a72955018595fc945e9410ae59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecf220a72955018595fc945e9410ae59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_4cb9ca95528e2061ba7ada35a9e930ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([4141, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4141, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a65c6c828c037e37fc9b12d1d0ba6959(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.13160091638565063, 0.15993918478488922, 0.49069681763648987, 0.04086945950984955], [0.44929584860801697, 0.12802210450172424, 0.37818604707717896, 0.38433870673179626], [0.31648457050323486, 0.3652629852294922, 0.01256631501019001, 0.024957232177257538], [0.13160091638565063, 0.15993918478488922, 0.49069681763648987, 0.04086945950984955], [0.4873030483722687, 0.08335397392511368, 0.12069840729236603, 0.12630900740623474], [0.3815208077430725, 0.08395334333181381, 0.4350278079509735, 0.0954495370388031], [0.4873030483722687, 0.08335397392511368, 0.12069840729236603, 0.12630900740623474]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.27369481325149536, 0.22894294559955597, 0.23795215785503387, 0.23634284734725952], [0.4143446683883667, 0.045904990285634995, 0.21750763058662415, 0.04185007885098457], [0.3132628798484802, 0.11117227375507355, 0.06890670955181122, 0.2404709905385971], [0.27369481325149536, 0.22894294559955597, 0.23795215785503387, 0.23634284734725952], [0.001952070975676179, 0.03246507793664932, 0.06659835577011108, 0.012370835989713669], [0.4863772392272949, 0.023673398420214653, 0.0706712156534195, 0.15412107110023499], [0.001952070975676179, 0.03246507793664932, 0.06659835577011108, 0.012370835989713669]], dtype='float32').reshape([7, 4]),
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