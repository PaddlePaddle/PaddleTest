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


class TestPrimitiveOp_b5004686d83a0af15cdc53fa503549bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.1617749035358429]], [[0.2343984693288803]], [[0.3363434672355652]], [[0.43660199642181396]], [[0.22570668160915375]], [[0.4208779036998749]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.5973773002624512]], [[0.6169924736022949]], [[0.8154162764549255]], [[0.6751572489738464]], [[0.7172725200653076]], [[0.797681450843811]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_3738d1d5e4d3e84466cdfe26ae4bc7e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3767072856426239]], [[0.013013598509132862]], [[0.31021395325660706]], [[0.2536526322364807]], [[0.4243045747280121]], [[0.35296210646629333]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.5493918657302856]], [[0.8088148236274719]], [[0.524810254573822]], [[0.6945334672927856]], [[0.6179171800613403]], [[0.6792595386505127]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_aca0c08b2632c61d72b33e5250a6839e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.2955186367034912, 0.26068830490112305]], [[0.023479899391531944, 0.39427757263183594]], [[0.3154904246330261, 0.29169759154319763]], [[0.3713768422603607, 0.4574488699436188]], [[0.24970504641532898, 0.20455273985862732]], [[0.31838396191596985, 0.1830456405878067]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.33132031559944153, 0.2283240407705307]], [[0.2721134126186371, 0.36326339840888977]], [[0.24461989104747772, 0.07305268943309784]], [[0.05029895529150963, 0.45751121640205383]], [[0.17434179782867432, 0.4313022792339325]], [[0.44769278168678284, 0.4641532897949219]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_89fb2a61ee23f588bc55f5fc1cccce97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.03718414902687073, 0.4314892888069153]], [[0.31938284635543823, 0.41265636682510376]], [[0.20889650285243988, 0.1257062703371048]], [[0.02582596056163311, 0.08291486650705338]], [[0.32461312413215637, 0.340871125459671]], [[0.45930325984954834, 0.21028712391853333]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.33132031559944153, 0.2283240407705307]], [[0.2721134126186371, 0.36326339840888977]], [[0.24461989104747772, 0.07305268943309784]], [[0.05029895529150963, 0.45751121640205383]], [[0.17434179782867432, 0.4313022792339325]], [[0.44769278168678284, 0.4641532897949219]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class TestPrimitiveOp_dfe3015e1f191c909ac700f466d30dac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b0071a67648dd3072ffe70503722fe9
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.49721720814704895, 0.058648962527513504]], [[0.17420677840709686, 0.2730419933795929]], [[0.12122957408428192, 0.16298291087150574]], [[0.02071063406765461, 0.09233691543340683]], [[0.296882688999176, 0.1320924460887909]], [[0.3168221116065979, 0.38519495725631714]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class TestPrimitiveOp_e49ff24603c1c4427ab2773810ad53dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
            paddle.to_tensor([0.16581030189990997, 0.14193306863307953, 0.008114946074783802, 0.11502566188573837, 0.2499755322933197, 0.23518739640712738, 0.4792259931564331, 0.4046315550804138, 0.2663853168487549, 0.13719098269939423, 0.2891804873943329, 0.33244985342025757, 0.3662383258342743, 0.39133161306381226, 0.13244225084781647, 0.39501598477363586], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_f8b1e7ae25294750e305cc82d2086a4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.16581030189990997, 0.14193306863307953, 0.008114946074783802, 0.11502566188573837, 0.2499755322933197, 0.23518739640712738, 0.4792259931564331, 0.4046315550804138, 0.2663853168487549, 0.13719098269939423, 0.2891804873943329, 0.33244985342025757, 0.3662383258342743, 0.39133161306381226, 0.13244225084781647, 0.39501598477363586], dtype='float32').reshape([16]),
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


class TestPrimitiveOp_6fc035db62b836965d5ef87c2b15786b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1756, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1756, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f767a14c7afa0bb2ce22a803b2bb2a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f767a14c7afa0bb2ce22a803b2bb2a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f767a14c7afa0bb2ce22a803b2bb2a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f767a14c7afa0bb2ce22a803b2bb2a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f767a14c7afa0bb2ce22a803b2bb2a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f767a14c7afa0bb2ce22a803b2bb2a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f767a14c7afa0bb2ce22a803b2bb2a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f767a14c7afa0bb2ce22a803b2bb2a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f767a14c7afa0bb2ce22a803b2bb2a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f767a14c7afa0bb2ce22a803b2bb2a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f767a14c7afa0bb2ce22a803b2bb2a94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1756, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_6fc035db62b836965d5ef87c2b15786b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1756, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1756, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_125170853725b0ee9469b61c13628a19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19554202258586884, 0.4953429400920868, 0.43110334873199463, 0.28431496024131775], [0.044859640300273895, 0.2850659489631653, 0.201993927359581, 0.20804829895496368], [0.4824425280094147, 0.029384635388851166, 0.45882564783096313, 0.007176108658313751], [0.2902381420135498, 0.2796521484851837, 0.3312593102455139, 0.07550826668739319], [0.055321916937828064, 0.4946301579475403, 0.174210786819458, 0.1292206197977066]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.38524216413497925, 0.27388086915016174, 0.45546597242355347, 0.49538132548332214], [0.3622490167617798, 0.414306640625, 0.14941883087158203, 0.24365560710430145], [0.3192705512046814, 0.18432563543319702, 0.324616938829422, 0.295026570558548], [0.42816928029060364, 0.2268391251564026, 0.13494037091732025, 0.30142292380332947], [0.235317662358284, 0.2238970249891281, 0.11962761729955673, 0.2108353078365326]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_0d183f9bd0d82a1e5e7ec86ac74b1484(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17329902946949005, 0.37573227286338806, 0.0912601426243782, 0.2594345211982727], [0.4604780972003937, 0.13689404726028442, 0.33471086621284485, 0.40724828839302063], [0.3390353322029114, 0.3622819781303406, 0.47980403900146484, 0.42228075861930847], [0.4604780972003937, 0.13689404726028442, 0.33471086621284485, 0.40724828839302063], [0.3390353322029114, 0.3622819781303406, 0.47980403900146484, 0.42228075861930847]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.031263820827007294, 0.10269512236118317, 0.10125775635242462, 0.11892905086278915], [0.22092019021511078, 0.33530178666114807, 0.20405182242393494, 0.41780561208724976], [0.15210598707199097, 0.35841765999794006, 0.38705116510391235, 0.42461442947387695], [0.22092019021511078, 0.33530178666114807, 0.20405182242393494, 0.41780561208724976], [0.15210598707199097, 0.35841765999794006, 0.38705116510391235, 0.42461442947387695]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_c25caa795026c4df4b14ebbd43544089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3370451331138611], [0.025026828050613403], [0.04696405306458473], [0.1218668669462204], [0.15391626954078674], [0.050961192697286606], [0.30239981412887573], [0.23399347066879272], [0.21992023289203644]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.4907219707965851], [0.47655177116394043], [0.45146989822387695], [0.37152010202407837], [0.4038541316986084], [0.4664810597896576], [0.2811638414859772], [0.3452494740486145], [0.3210054337978363]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_a8ea17bd2b96d40bcf443cc29e8316cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03724591061472893], [0.011372104287147522], [0.10467681288719177], [0.13299866020679474], [0.3006611168384552], [0.09304334223270416], [0.3312247097492218], [0.11871779710054398], [0.08097352087497711]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.4733383357524872], [0.20966672897338867], [0.15613777935504913], [0.4914016127586365], [0.4056450426578522], [0.43923839926719666], [0.43668332695961], [0.4740495979785919], [0.0606311671435833]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_2f7360b2582777258f33c5504c94e869(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3370451331138611], [0.025026828050613403], [0.4009562134742737], [0.37868067622184753], [0.15391626954078674], [0.050961192697286606], [0.3595850467681885], [0.23399347066879272], [0.21992023289203644]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.4907219707965851], [0.47655177116394043], [0.45146989822387695], [0.2179049253463745], [0.2145148366689682], [0.42308953404426575], [0.2811638414859772], [0.3096412420272827], [0.08616721630096436]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_202e00c4e6da502b5848e67ce32ee290(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03724591061472893], [0.011372104287147522], [0.10467681288719177], [0.45974865555763245], [0.47817090153694153], [0.09304334223270416], [0.3312247097492218], [0.3817911446094513], [0.08097352087497711]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.4733383357524872], [0.09571289271116257], [0.15195058286190033], [0.012989894486963749], [0.4056450426578522], [0.43923839926719666], [0.357232004404068], [0.16211768984794617], [0.025704603642225266]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_2e22aea19351e082b8deb85d0639dde2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.34383171796798706], [0.27951961755752563], [0.04696405306458473], [0.1218668669462204], [0.46935001015663147], [0.4291514456272125], [0.30239981412887573], [0.36811432242393494], [0.2899019420146942]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.4125475287437439], [0.2618580162525177], [0.14666202664375305], [0.37152010202407837], [0.4038541316986084], [0.4664810597896576], [0.22648189961910248], [0.3452494740486145], [0.3210054337978363]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_be06623fa93a41dfc42852995b26f302(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3704785704612732], [0.022056521847844124], [0.3112718164920807], [0.13299866020679474], [0.3006611168384552], [0.48700836300849915], [0.43817874789237976], [0.11871779710054398], [0.35156527161598206]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.4564463198184967], [0.20966672897338867], [0.15613777935504913], [0.4914016127586365], [0.3907115161418915], [0.38277748227119446], [0.43668332695961], [0.4740495979785919], [0.0606311671435833]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_a0840ec5d123642746c4ef09a9c0ae50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0729246512055397], [0.034768473356962204], [-0.013078576885163784], [0.16130442917346954], [-0.010292893275618553], [0.12493808567523956], [-0.0019259940600022674], [-0.02474241331219673], [-0.0016566826961934566]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_000e1959cd328aaf6af75692acadfe23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.34383171796798706], [0.27951961755752563], [0.4009562134742737], [0.37868067622184753], [0.46935001015663147], [0.4291514456272125], [0.3595850467681885], [0.36811432242393494], [0.2899019420146942]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.4125475287437439], [0.2618580162525177], [0.14666202664375305], [0.2179049253463745], [0.2145148366689682], [0.42308953404426575], [0.22648189961910248], [0.3096412420272827], [0.08616721630096436]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_f86cebebc989e6ed2846e9e189de19fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3704785704612732], [0.022056521847844124], [0.3112718164920807], [0.45974865555763245], [0.47817090153694153], [0.48700836300849915], [0.43817874789237976], [0.3817911446094513], [0.35156527161598206]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.4564463198184967], [0.09571289271116257], [0.15195058286190033], [0.012989894486963749], [0.3907115161418915], [0.38277748227119446], [0.357232004404068], [0.16211768984794617], [0.025704603642225266]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_81044d8e314b7a7e64295811258946c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.005907343700528145], [-0.0013008894165977836], [0.04051446169614792], [0.0718279704451561], [0.022287728264927864], [0.0006318385130725801], [0.010774265974760056], [0.012844983488321304], [0.06638913601636887]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0729246512055397], [0.034768473356962204], [-0.013078576885163784], [0.16130442917346954], [-0.010292893275618553], [0.12493808567523956], [-0.0019259939435869455], [-0.02474241331219673], [-0.0016566825797781348]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_6af9f6581cad698e5c85c71f632f1a24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [-0.0], [0.0], [-0.0], [0.0], [-0.0], [-0.0], [-0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-11.344745635986328], [27.726694107055664], [1.322812557220459], [-1.245705008506775], [1.461818814277649], [-196.73736572265625], [1.1787587404251099], [2.926231622695923], [1.024954080581665]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_b683a7538b123f277ffef55f48129604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a523845849f255fc7ef49c9c0837b9b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.43122604489326477]], [[0.1552562713623047]], [[0.06082388386130333]], [[0.4877679944038391]], [[0.0408121794462204]], [[0.12468642741441727]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.5539252161979675]], [[0.6346803307533264]], [[0.7276504635810852]], [[0.5098108649253845]], [[0.7054385542869568]], [[0.6356890797615051]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_a63cc6c2e6b5c28bbfbe7d96302072c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.24540750682353973]], [[0.034014638513326645]], [[0.34575676918029785]], [[0.43839043378829956]], [[0.4028033912181854]], [[0.2140284925699234]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.5134385228157043]], [[0.6661975979804993]], [[0.6664236187934875]], [[0.6304938793182373]], [[0.7162249088287354]], [[0.5771429538726807]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_ea164ead1e8dabe81b2748a633c9a775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([5551, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5551, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cfbbdb1a5e802c07f1ce624272576e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cfbbdb1a5e802c07f1ce624272576e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cfbbdb1a5e802c07f1ce624272576e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cfbbdb1a5e802c07f1ce624272576e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cfbbdb1a5e802c07f1ce624272576e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cfbbdb1a5e802c07f1ce624272576e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cfbbdb1a5e802c07f1ce624272576e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cfbbdb1a5e802c07f1ce624272576e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cfbbdb1a5e802c07f1ce624272576e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cfbbdb1a5e802c07f1ce624272576e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cfbbdb1a5e802c07f1ce624272576e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5551, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_ea164ead1e8dabe81b2748a633c9a775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([5551, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5551, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54e479cbea0fbca7581042b9c21dc969(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33677515387535095, 0.30652493238449097, 0.367003858089447, 0.20469045639038086], [0.10388068109750748, 0.12189987301826477, 0.2237544059753418, 0.1381952166557312], [0.4053584933280945, 0.09487839043140411, 0.4053238034248352, 0.43379074335098267], [0.10388068109750748, 0.12189987301826477, 0.2237544059753418, 0.1381952166557312], [0.4053584933280945, 0.09487839043140411, 0.4053238034248352, 0.43379074335098267], [0.0876406729221344, 0.3536801338195801, 0.027936020866036415, 0.2873351275920868], [0.0876406729221344, 0.3536801338195801, 0.027936020866036415, 0.2873351275920868]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.134425088763237, 0.2605721056461334, 0.14654327929019928, 0.13377103209495544], [0.07293681800365448, 0.27893778681755066, 0.051850758492946625, 0.29999396204948425], [0.11252158135175705, 0.12145049124956131, 0.38858047127723694, 0.3132563531398773], [0.07293681800365448, 0.27893778681755066, 0.051850758492946625, 0.29999396204948425], [0.11252158135175705, 0.12145049124956131, 0.38858047127723694, 0.3132563531398773], [0.3163917362689972, 0.161368727684021, 0.027262717485427856, 0.30870914459228516], [0.3163917362689972, 0.161368727684021, 0.027262717485427856, 0.30870914459228516]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_4b7905db4233b8124cafc84d4f5e63ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4078967571258545, 0.37209582328796387, 0.23028191924095154, 0.16957210004329681, 0.18082909286022186, 0.4491502642631531], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3997354805469513, 0.39295482635498047, 0.09432562440633774, 0.49924978613853455, 0.30787521600723267, 0.35715749859809875], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_c1ccc46e4c1a9d055a4dfbc87de96154(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18946067988872528, 0.18211600184440613, 0.004157352261245251, 0.01153471041470766, 0.360180139541626, 0.3169853091239929], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4620789587497711, 0.2747343182563782, 0.18587720394134521, 0.3222280740737915, 0.2506195902824402, 0.2472265213727951], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_39099b45553ed6de8077facec2d54c47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.056108973920345306, 0.36671507358551025, 0.13837383687496185, 0.014441891573369503, 0.34063658118247986, 0.30013731122016907], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0959656834602356, 0.010401327162981033, 0.11267633736133575, 0.08521786332130432, 0.09726300835609436, 0.05800497531890869], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ef6e99b41c92bce33ec6731b43f2ea06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18877169489860535, 0.04359738901257515, 0.3416917026042938, 0.2778036594390869, 0.22354553639888763, 0.17368389666080475], dtype='float32').reshape([6]),
            paddle.to_tensor([0.08197572827339172, 0.33816057443618774, 0.29246312379837036, 0.4799637198448181, 0.3316073417663574, 0.26939353346824646], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_1b9a1495ae245f5e848f13b2c7210827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.056108973920345306, 0.36671507358551025, 0.13837383687496185, 0.014441891573369503, 0.30787521600723267, 0.30013731122016907], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3997354805469513, 0.39295482635498047, 0.11267633736133575, 0.49924978613853455, 0.30787521600723267, 0.35715749859809875], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_067e037f17044f8362453100764d90eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18877169489860535, 0.04359738901257515, 0.18587720394134521, 0.2778036594390869, 0.22354553639888763, 0.17368389666080475], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4620789587497711, 0.33816057443618774, 0.29246312379837036, 0.4799637198448181, 0.3316073417663574, 0.26939353346824646], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ff4b960659020c3c5704a6c8e09123e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4078967571258545, 0.39295482635498047, 0.23028191924095154, 0.49924978613853455, 0.30787521600723267, 0.4491502642631531], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3997354805469513, 0.39295482635498047, 0.09432562440633774, 0.49924978613853455, 0.30787521600723267, 0.35715749859809875], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_78a65c2eee8404fed4ff36a1dd6c22d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4620789587497711, 0.2747343182563782, 0.18587720394134521, 0.3222280740737915, 0.360180139541626, 0.3169853091239929], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4620789587497711, 0.2747343182563782, 0.18587720394134521, 0.3222280740737915, 0.2506195902824402, 0.2472265213727951], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_84b1029330fcf8e12efd05fae7803457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.004256535787135363, -0.10495690256357193, 0.001265051425434649, 0.014308074489235878, -0.026299387216567993, -0.016757093369960785], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, 0.0, -0.0, 0.0, -0.0, 0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_bb02f05ecb3b9288ac4529c8de24c829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4038161039352417, 0.38252532482147217, 0.16230377554893494, 0.3344109356403351, 0.24435216188430786, 0.4031538963317871], dtype='float32').reshape([6]),
            paddle.to_tensor([0.07603733241558075, 0.1885582059621811, 0.1255250871181488, 0.0498298779129982, 0.2189497947692871, 0.17907114326953888], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ac395b73a4aa1505f1d758da12668c06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3257698118686676, 0.22842516005039215, 0.09501727670431137, 0.16688139736652374, 0.3053998649120331, 0.2821059226989746], dtype='float32').reshape([6]),
            paddle.to_tensor([0.13537371158599854, 0.1908789873123169, 0.3170773983001709, 0.3788836896419525, 0.2775764465332031, 0.2215387225151062], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_2f325d16a0d00d77ad952beceb7c317f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4078967571258545, 0.39295482635498047, 0.23028191924095154, 0.49924978613853455, 0.34063658118247986, 0.4491502642631531], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0959656834602356, 0.010401327162981033, 0.09432562440633774, 0.08521786332130432, 0.09726300835609436, 0.05800497531890869], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_b9db06133ca80f8e269fe64440763fa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4620789587497711, 0.2747343182563782, 0.3416917026042938, 0.3222280740737915, 0.360180139541626, 0.3169853091239929], dtype='float32').reshape([6]),
            paddle.to_tensor([0.08197572827339172, 0.2747343182563782, 0.18587720394134521, 0.3222280740737915, 0.2506195902824402, 0.2472265213727951], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_2f5b7ec9357da76c63c702c592f29602(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3571954071521759, -0.8799879550933838, 0.48109522461891174, 0.3367627263069153, -1.1529297828674316, -1.1943671703338623], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.029927704483270645, 0.22151875495910645, -0.6423251628875732, 0.8150354027748108, -0.8591655492782593, 0.9219996929168701], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_e65aa6d07b46e593b420b7a03555ce40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1769, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1769, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb8cb7360aaca624e75a90eedca5c2fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb8cb7360aaca624e75a90eedca5c2fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb8cb7360aaca624e75a90eedca5c2fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb8cb7360aaca624e75a90eedca5c2fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb8cb7360aaca624e75a90eedca5c2fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb8cb7360aaca624e75a90eedca5c2fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb8cb7360aaca624e75a90eedca5c2fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb8cb7360aaca624e75a90eedca5c2fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb8cb7360aaca624e75a90eedca5c2fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb8cb7360aaca624e75a90eedca5c2fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb8cb7360aaca624e75a90eedca5c2fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1769, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_e65aa6d07b46e593b420b7a03555ce40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1769, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1769, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f93bd446c350bd03f6f8493cc8c52e9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4976271390914917, 0.49329546093940735, 0.38484594225883484, 0.36033523082733154, 0.18437574803829193, 0.09304941445589066, 0.032206643372774124, 0.0766223669052124, 0.43633055686950684, 0.12222424894571304, 0.2502988576889038, 0.020448602735996246, 0.2181742787361145, 0.34769222140312195, 0.1660960465669632, 0.08530177175998688, 0.08622681349515915, 0.47288262844085693, 0.04534696042537689, 0.3430462181568146, 0.22856423258781433, 0.46614816784858704, 0.4240947365760803, 0.259934663772583], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_d3d46f3f1cc9a4e6ca42de3da2e73b0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4976271390914917, 0.49329546093940735, 0.38484594225883484, 0.36033523082733154, 0.18437574803829193, 0.09304941445589066, 0.032206643372774124, 0.0766223669052124, 0.43633055686950684, 0.12222424894571304, 0.2502988576889038, 0.020448602735996246, 0.2181742787361145, 0.34769222140312195, 0.1660960465669632, 0.08530177175998688, 0.08622681349515915, 0.47288262844085693, 0.04534696042537689, 0.3430462181568146, 0.22856423258781433, 0.46614816784858704, 0.4240947365760803, 0.259934663772583], dtype='float32').reshape([24]),
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


class TestPrimitiveOp_031202dd1862c0c808a8f621b935fa7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1502, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1502, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf2255597764918fcb70f09bcfe2f275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf2255597764918fcb70f09bcfe2f275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf2255597764918fcb70f09bcfe2f275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf2255597764918fcb70f09bcfe2f275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf2255597764918fcb70f09bcfe2f275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf2255597764918fcb70f09bcfe2f275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf2255597764918fcb70f09bcfe2f275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf2255597764918fcb70f09bcfe2f275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf2255597764918fcb70f09bcfe2f275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf2255597764918fcb70f09bcfe2f275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf2255597764918fcb70f09bcfe2f275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1502, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_031202dd1862c0c808a8f621b935fa7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1502, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1502, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_d6e2aa9c2c70bebc1fa0cef893b2db9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
            paddle.to_tensor([0.05023837462067604, 0.17656466364860535, 0.4871646761894226, 0.25723910331726074], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_0c277bfc5126847f3f29c5b11cd0e66d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05023837462067604, 0.17656466364860535, 0.4871646761894226, 0.25723910331726074], dtype='float32').reshape([4]),
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


class TestPrimitiveOp_e0c7327985ed3d95c0a8bddca1cd054b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20413066446781158, 0.12532176077365875, 0.28872770071029663, 0.07178770750761032], [0.21112103760242462, 0.2790716588497162, 0.38941124081611633, 0.05386298522353172], [0.008499457500874996, 0.043766867369413376, 0.11801748722791672, 0.03665297105908394], [0.17231427133083344, 0.3661665618419647, 0.3355735242366791, 0.011698858812451363], [0.17231427133083344, 0.3661665618419647, 0.3355735242366791, 0.011698858812451363], [0.008499457500874996, 0.043766867369413376, 0.11801748722791672, 0.03665297105908394]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.4314934015274048, 0.4877358376979828, 0.29400381445884705, 0.29048416018486023], [0.19290420413017273, 0.13745276629924774, 0.21723920106887817, 0.02726193703711033], [0.0250072181224823, 0.16611385345458984, 0.23217295110225677, 0.21143263578414917], [0.4995115101337433, 0.25204506516456604, 0.41638800501823425, 0.03359423205256462], [0.4995115101337433, 0.25204506516456604, 0.41638800501823425, 0.03359423205256462], [0.0250072181224823, 0.16611385345458984, 0.23217295110225677, 0.21143263578414917]], dtype='float32').reshape([6, 4]),
        ]


class TestPrimitiveOp_eec296e47e887fbade66b3eaeb461fbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.037359751760959625, 0.4428909718990326, 0.025828247889876366, 0.07863251864910126], [0.414345920085907, 0.37859612703323364, 0.40324094891548157, 0.08170878142118454], [0.4363548159599304, 0.3131655752658844, 0.2752301096916199, 0.10653684288263321], [0.21835356950759888, 0.16582512855529785, 0.015149720013141632, 0.01305893063545227], [0.037359751760959625, 0.4428909718990326, 0.025828247889876366, 0.07863251864910126]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.30960986018180847, 0.2964046001434326, 0.30743059515953064, 0.3093737065792084], [0.45385125279426575, 0.0017529912292957306, 0.47665098309516907, 0.04158458486199379], [0.15653115510940552, 0.2814658284187317, 0.2737646996974945, 0.25372228026390076], [0.33385902643203735, 0.2507376968860626, 0.21649283170700073, 0.3594752550125122], [0.30960986018180847, 0.2964046001434326, 0.30743059515953064, 0.3093737065792084]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_dc38fbca4d28435e70b9ea4f76b0680d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ccbf9da3f704217b2af065cf01069cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1071244403719902]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.15802021324634552]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_d79aa6cbe178af236a246c2fc63cd304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.32788488268852234]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.2992727756500244]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_1add03bc4d75e6855a839950e379cda8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3475726544857025]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.15802021324634552]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_209b3df9c48890d3e839597d6a69aaf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.47082066535949707]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.288260281085968]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_e4961679e0b64fd8343275f6a0e1f17b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1071244403719902]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.028114398941397667]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_d79aa6cbe178af236a246c2fc63cd304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.32788488268852234]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.2992727756500244]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_5414383d9940a8561b451b261184d11f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03686540946364403]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_de63f6a8e809cd18b1bcbc2aeede9a8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3475726544857025]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.028114398941397667]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_209b3df9c48890d3e839597d6a69aaf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.47082066535949707]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.288260281085968]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_efeab80d928578ace774d73dc10b8390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05832042172551155]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.03686540946364403]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_e5967fdbbc3871171754f66357ebbacb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3678816258907318]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_bd5c2a895566f9c917ca022b5d28a9ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19522464275360107], [0.29033544659614563], [0.25756993889808655], [0.002446115715429187], [0.14687369763851166], [0.015598968602716923]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.41157740354537964], [0.4826880991458893], [0.4846378266811371], [0.25960448384284973], [0.16274337470531464], [0.08710668981075287]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_dd785bcc943bfffc935f53d1fd769a04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.016921646893024445], [0.11776749044656754], [0.10755512863397598], [0.23614223301410675], [0.055858783423900604], [0.2302117496728897]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.4798603653907776], [0.32325857877731323], [0.2651921510696411], [0.1680978387594223], [0.03530534356832504], [0.18742431700229645]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_30a851d0034a839425a3bb32258be56b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19522464275360107], [0.29033544659614563], [0.25756993889808655], [0.4750064015388489], [0.4037943184375763], [0.015598968602716923]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.41157740354537964], [0.4826880991458893], [0.4846378266811371], [0.25960448384284973], [0.1073368713259697], [0.08710668981075287]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_3dda508979e7c359321ffb6908e2d61f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.498967707157135], [0.11776749044656754], [0.10755512863397598], [0.23614223301410675], [0.1757691651582718], [0.31581953167915344]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.4798603653907776], [0.32325857877731323], [0.02578026056289673], [0.1680978387594223], [0.018725527450442314], [0.12951743602752686]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_326f2e65cc33492e8fd1c00059af1552(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2858388125896454], [0.3007555902004242], [0.3974524438381195], [0.002446115715429187], [0.14687369763851166], [0.48988133668899536]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3591238260269165], [0.3641086220741272], [0.1448068916797638], [0.0998174175620079], [0.16274337470531464], [0.04630977287888527]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_10e1f3727331c5cbb95fe72bd56a06e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.016921646893024445], [0.35783061385154724], [0.12393762171268463], [0.2560504674911499], [0.055858783423900604], [0.2302117496728897]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.22245408594608307], [0.08651915937662125], [0.2651921510696411], [0.07728420943021774], [0.03530534356832504], [0.18742431700229645]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_566902ce4d9db894bb029ce4ef92b25a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.010928520932793617], [0.022338353097438812], [-0.05425577610731125], [-0.0027498090639710426], [0.046230580657720566], [0.005657250061631203]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_a0cd4700c79f003e5eadf5d744dcbc2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2858388125896454], [0.3007555902004242], [0.3974524438381195], [0.4750064015388489], [0.4037943184375763], [0.48988133668899536]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3591238260269165], [0.3641086220741272], [0.1448068916797638], [0.0998174175620079], [0.1073368713259697], [0.04630977287888527]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_5577b7803d4bf55399f770437b0b038b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.498967707157135], [0.35783061385154724], [0.12393762171268463], [0.2560504674911499], [0.1757691651582718], [0.31581953167915344]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.22245408594608307], [0.08651915937662125], [0.02578026056289673], [0.07728420943021774], [0.018725527450442314], [0.12951743602752686]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_f3491b24dd4bfcb1b267628be400aa86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.02026430517435074], [-0.017188403755426407], [0.02479902096092701], [0.06707112491130829], [0.046556755900382996], [0.08263831585645676]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.010928520932793617], [0.022338353097438812], [-0.05425577610731125], [-0.0027498090639710426], [0.046230580657720566], [0.005657250061631203]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_2b153de226924356a060f95995d5d421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [-0.0], [-0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[1.5392990112304688], [2.2996177673339844], [3.187819242477417], [1.0409983396530151], [0.0070059699937701225], [0.9315420389175415]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_518c638aaa9867770e2ec9a79c491fae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08269330859184265, 0.15934213995933533, 0.11298678070306778, 0.3357259929180145], [0.002818088512867689, 0.23228012025356293, 0.18485650420188904, 0.3241109848022461], [0.06744459271430969, 0.24970084428787231, 0.3579083979129791, 0.4362775683403015], [0.026034051552414894, 0.4280358552932739, 0.45209744572639465, 0.24006299674510956]], dtype='float32').reshape([4, 4]),
            paddle.to_tensor([[0.33489540219306946, 0.18969909846782684, 0.15051589906215668, 0.4941870868206024], [0.477124959230423, 0.1238592341542244, 0.3813120126724243, 0.020048359408974648], [0.30283045768737793, 0.06579912453889847, 0.22384823858737946, 0.44903630018234253], [0.14519372582435608, 0.059971101582050323, 0.11059994250535965, 0.11827445775270462]], dtype='float32').reshape([4, 4]),
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


class TestPrimitiveOp_6a0f588775016c25d4cceb99ad7eb572(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2080, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2080, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6faaac000b54c3c21a1392cf91fa64c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6faaac000b54c3c21a1392cf91fa64c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6faaac000b54c3c21a1392cf91fa64c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6faaac000b54c3c21a1392cf91fa64c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6faaac000b54c3c21a1392cf91fa64c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6faaac000b54c3c21a1392cf91fa64c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6faaac000b54c3c21a1392cf91fa64c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6faaac000b54c3c21a1392cf91fa64c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6faaac000b54c3c21a1392cf91fa64c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6faaac000b54c3c21a1392cf91fa64c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6faaac000b54c3c21a1392cf91fa64c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2080, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_6a0f588775016c25d4cceb99ad7eb572(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2080, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2080, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_006b4b9e96949cb55704cb3ad82d80fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.24348896741867065, 0.3023703098297119, 0.12081083655357361, 0.3574257791042328], [0.24348896741867065, 0.3023703098297119, 0.12081083655357361, 0.3574257791042328], [0.10204478353261948, 0.1296604424715042, 0.3511778712272644, 0.37410691380500793], [0.3333530128002167, 0.38378939032554626, 0.21012456715106964, 0.22895927727222443], [0.18427875638008118, 0.06358519941568375, 0.4557211101055145, 0.1475222110748291], [0.3554903268814087, 0.374887079000473, 0.16366685926914215, 0.25883519649505615], [0.38285961747169495, 0.15902747213840485, 0.26753944158554077, 0.22846563160419464]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.45793333649635315, 0.3819330036640167, 0.19847269356250763, 0.11775033921003342], [0.45793333649635315, 0.3819330036640167, 0.19847269356250763, 0.11775033921003342], [0.11242612451314926, 0.3438863754272461, 0.27541983127593994, 0.4314340054988861], [0.2678462862968445, 0.17077229917049408, 0.38100945949554443, 0.3379964530467987], [0.32448121905326843, 0.11712616682052612, 0.4017871022224426, 0.07933226227760315], [0.15229831635951996, 0.03220883011817932, 0.46974676847457886, 0.4247609078884125], [0.16722066700458527, 0.3433651328086853, 0.49674180150032043, 0.22078944742679596]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_f543955f39482d4e5d31190750d55247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([4585, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4585, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eace2e65f371c73475bba9df1dd68f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eace2e65f371c73475bba9df1dd68f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eace2e65f371c73475bba9df1dd68f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eace2e65f371c73475bba9df1dd68f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eace2e65f371c73475bba9df1dd68f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eace2e65f371c73475bba9df1dd68f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eace2e65f371c73475bba9df1dd68f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eace2e65f371c73475bba9df1dd68f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eace2e65f371c73475bba9df1dd68f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eace2e65f371c73475bba9df1dd68f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eace2e65f371c73475bba9df1dd68f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4585, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f543955f39482d4e5d31190750d55247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([4585, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4585, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23768db297fb5636c2a5625dacea4379(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1048, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1048, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d075d27f1006327bacd2533af51cb299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d075d27f1006327bacd2533af51cb299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d075d27f1006327bacd2533af51cb299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d075d27f1006327bacd2533af51cb299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d075d27f1006327bacd2533af51cb299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d075d27f1006327bacd2533af51cb299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d075d27f1006327bacd2533af51cb299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d075d27f1006327bacd2533af51cb299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d075d27f1006327bacd2533af51cb299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d075d27f1006327bacd2533af51cb299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d075d27f1006327bacd2533af51cb299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1048, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_23768db297fb5636c2a5625dacea4379(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1048, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1048, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c9460cd58abff38d75364249aa3e12c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_24fda1e13a632e2cb00e0b6e547de181(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.24996156990528107, 0.22913239896297455, 0.4686241149902344, 0.3902522325515747], [0.19338367879390717, 0.49108168482780457, 0.3728701174259186, 0.21134459972381592], [0.19338367879390717, 0.49108168482780457, 0.3728701174259186, 0.21134459972381592], [0.3119868338108063, 0.1919063776731491, 0.09895597398281097, 0.19015774130821228], [0.49074211716651917, 0.14704328775405884, 0.07624967396259308, 0.2689438760280609], [0.3642599880695343, 0.4556441605091095, 0.09806236624717712, 0.2138247936964035]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.0037596742622554302, 0.30459901690483093, 0.027924980968236923, 0.08544733375310898], [0.14154160022735596, 0.11170454323291779, 0.10085401684045792, 0.48489460349082947], [0.14154160022735596, 0.11170454323291779, 0.10085401684045792, 0.48489460349082947], [0.2508198022842407, 0.19932301342487335, 0.27499574422836304, 0.4788191318511963], [0.03897927701473236, 0.0643027052283287, 0.09212637692689896, 0.21963757276535034], [0.3734019696712494, 0.16025450825691223, 0.010875063017010689, 0.07821814715862274]], dtype='float32').reshape([6, 4]),
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


class TestPrimitiveOp_c156cfe699732e9fcdaf9512374a761f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47492f828e721b67b39f9215d060228c
    def get_inputs(self):
        return [
            paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.6485222578048706, 0.3337302505970001, 0.9457365870475769, 1.287784218788147], [0.09747067838907242, 0.2514309883117676, 0.4891703426837921, 1.1247966289520264]], dtype='float32').reshape([2, 4]),
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


class TestPrimitiveOp_a3a5974acc1ffa54baa9f9034d85fd12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ff0558d29e4c36c82c5fce402c13bdb
    def get_inputs(self):
        return [
            paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.11098602414131165, 0.5434260964393616, 0.8391168713569641, 2.329698085784912], [1.631260871887207, 1.9401860237121582, 3.489945888519287, 0.3373526632785797]], dtype='float32').reshape([2, 4]),
        ]


class TestPrimitiveOp_6ea4accd3d4398eb72fe09f1b6de518a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.13977095484733582], [0.19764673709869385], [0.020277973264455795], [0.3286098837852478], [0.13251619040966034]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.37055835127830505], [0.19263240694999695], [0.2666272819042206], [0.056447286158800125], [0.30655404925346375]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_05fab49d5533783ee279b61ff21f73e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.025554388761520386], [0.1028725877404213], [0.3705821931362152], [0.24572992324829102], [0.2983556091785431]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.29983288049697876], [0.2629903554916382], [0.1999022364616394], [0.3250402510166168], [0.24727387726306915]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_992384da892cbef87b1cdd20552fd3ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.48249489068984985], [0.19764673709869385], [0.020277973264455795], [0.3286098837852478], [0.24283719062805176]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.19752667844295502], [0.10475891828536987], [0.024959586560726166], [0.056447286158800125], [0.1750042885541916]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_8e63df1b959ceb159ed1dc71b141aea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.47020426392555237], [0.4614999294281006], [0.42352351546287537], [0.24572992324829102], [0.3263440430164337]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.07073542475700378], [0.2629903554916382], [0.18045572936534882], [0.3250402510166168], [0.24727387726306915]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_6b3bf124cca16aa25f670eb1ae28db6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.13977095484733582], [0.2654637098312378], [0.3343604803085327], [0.33652734756469727], [0.13251619040966034]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.37055835127830505], [0.19263240694999695], [0.2666272819042206], [0.045508455485105515], [0.30655404925346375]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_8eb082b8202688048ebd0367a5bf58e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.025554388761520386], [0.1028725877404213], [0.3705821931362152], [0.4677906334400177], [0.2983556091785431]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.29983288049697876], [0.19387735426425934], [0.1999022364616394], [0.3063305914402008], [0.06010952964425087]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_ae1b3506b6bc5952e17e09411fc29abb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17713594436645508], [0.01181112602353096], [0.010422749444842339], [0.025402620434761047], [-0.03610027953982353]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_7d8e287bef2dbb3ee44191df764eb537(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.48249489068984985], [0.2654637098312378], [0.3343604803085327], [0.33652734756469727], [0.24283719062805176]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.19752667844295502], [0.10475891828536987], [0.024959586560726166], [0.045508455485105515], [0.1750042885541916]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_85b066b4e01c69f52ec73d6bbe41ce33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.47020426392555237], [0.4614999294281006], [0.42352351546287537], [0.4677906334400177], [0.3263440430164337]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.07073542475700378], [0.19387735426425934], [0.18045572936534882], [0.3063305914402008], [0.06010952964425087]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_140befb82ebd7976e7a32e3407a5b277(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11383591592311859], [0.0430082343518734], [0.07520538568496704], [0.046987924724817276], [0.018059460446238518]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.17713594436645508], [0.01181112602353096], [0.010422749444842339], [0.025402620434761047], [-0.03610027953982353]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_00e1f85311f766de35324ba1b69315a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [-0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.5560637712478638], [0.725375235080719], [0.8614094853401184], [0.45937982201576233], [2.9989676475524902]], dtype='float32').reshape([5, 1]),
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


class TestPrimitiveOp_a55f72bc148ce2bbbbd47ccfd0402e55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2390, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2390, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09076e591277efa966d3fb6b7fa595ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09076e591277efa966d3fb6b7fa595ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09076e591277efa966d3fb6b7fa595ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09076e591277efa966d3fb6b7fa595ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09076e591277efa966d3fb6b7fa595ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09076e591277efa966d3fb6b7fa595ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09076e591277efa966d3fb6b7fa595ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09076e591277efa966d3fb6b7fa595ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09076e591277efa966d3fb6b7fa595ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09076e591277efa966d3fb6b7fa595ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09076e591277efa966d3fb6b7fa595ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2390, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_a55f72bc148ce2bbbbd47ccfd0402e55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2390, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2390, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_318fea8842b5a45eb4dc0557a488ab3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([3090, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3090, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fedeccf7d47cb7c30801038da6e55531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fedeccf7d47cb7c30801038da6e55531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fedeccf7d47cb7c30801038da6e55531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fedeccf7d47cb7c30801038da6e55531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fedeccf7d47cb7c30801038da6e55531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fedeccf7d47cb7c30801038da6e55531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fedeccf7d47cb7c30801038da6e55531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fedeccf7d47cb7c30801038da6e55531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fedeccf7d47cb7c30801038da6e55531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fedeccf7d47cb7c30801038da6e55531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fedeccf7d47cb7c30801038da6e55531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3090, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_318fea8842b5a45eb4dc0557a488ab3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([3090, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3090, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1170bd82db44fa62766a6ec6ed15f8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([3748, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3748, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_883812b15483baf4a01b31aa88d969ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_883812b15483baf4a01b31aa88d969ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_883812b15483baf4a01b31aa88d969ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_883812b15483baf4a01b31aa88d969ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_883812b15483baf4a01b31aa88d969ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_883812b15483baf4a01b31aa88d969ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_883812b15483baf4a01b31aa88d969ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_883812b15483baf4a01b31aa88d969ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_883812b15483baf4a01b31aa88d969ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_883812b15483baf4a01b31aa88d969ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_883812b15483baf4a01b31aa88d969ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3748, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f1170bd82db44fa62766a6ec6ed15f8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([3748, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3748, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_b06fe48423cb3cb02e127f419140af5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
            paddle.to_tensor([0.11505911499261856, 0.2667391300201416, 0.1605810821056366, 0.3698037564754486, 0.4984857439994812, 0.4004300832748413, 0.33577585220336914, 0.014704381115734577, 0.49942636489868164, 0.28690001368522644, 0.16109298169612885, 0.3741922676563263, 0.4992409646511078, 0.4941102862358093, 0.06382890045642853, 0.37459418177604675, 0.0037233552429825068, 0.0786420926451683, 0.14995817840099335, 0.14367970824241638], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_b33faa062b69b4d209efcd8d93f25995(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11505911499261856, 0.2667391300201416, 0.1605810821056366, 0.3698037564754486, 0.4984857439994812, 0.4004300832748413, 0.33577585220336914, 0.014704381115734577, 0.49942636489868164, 0.28690001368522644, 0.16109298169612885, 0.3741922676563263, 0.4992409646511078, 0.4941102862358093, 0.06382890045642853, 0.37459418177604675, 0.0037233552429825068, 0.0786420926451683, 0.14995817840099335, 0.14367970824241638], dtype='float32').reshape([20]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_296e0c10e1c7b1da934ff9a0bf471081(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.006397643592208624], [0.16135065257549286], [0.05680867284536362], [0.19531609117984772]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.37256571650505066], [0.37738490104675293], [0.40263471007347107], [0.16988332569599152]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_970c3f615d9e1e5dcccb1e8ee399e71d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.097120501101017], [0.03819771856069565], [0.08493614196777344], [0.2551638185977936]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4308514893054962], [0.21365579962730408], [0.3567523658275604], [0.47443392872810364]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_5e888e42e7df8ee572b5a6e70d5e5c7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.006397643592208624], [0.22528007626533508], [0.4595484733581543], [0.404490202665329]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.37155959010124207], [0.37738490104675293], [0.38180166482925415], [0.16988332569599152]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_5487c3b0c36bde6c715cad4332fe3035(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15801455080509186], [0.03819771856069565], [0.08493614196777344], [0.2796372175216675]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.35161545872688293], [0.1566532999277115], [0.24491369724273682], [0.47443392872810364]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_74baafe05975b05484abf13f46c774f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3430819809436798], [0.16135065257549286], [0.05680867284536362], [0.19531609117984772]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.37256571650505066], [0.25770851969718933], [0.40263471007347107], [0.15933819115161896]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_7291888d68a1cb57e064c30f1f2562bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.097120501101017], [0.3680274784564972], [0.0860719233751297], [0.2551638185977936]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4308514893054962], [0.21365579962730408], [0.3567523658275604], [0.20909146964550018]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_74209e0ea88f91530d7a368539fb7b04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08053532242774963], [0.003142738714814186], [0.08117058873176575], [-0.04404306039214134]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_4a0a9aa058f351b5544ac0b195825ac8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3430819809436798], [0.22528007626533508], [0.4595484733581543], [0.404490202665329]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.37155959010124207], [0.25770851969718933], [0.38180166482925415], [0.15933819115161896]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_fbe4f32c5e5e3cd062e836f56f456b4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15801455080509186], [0.3680274784564972], [0.0860719233751297], [0.2796372175216675]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.35161545872688293], [0.1566532999277115], [0.24491369724273682], [0.20909146964550018]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_93c20c1441befc7cdb6b6f61f64688aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.005513290874660015], [-0.006854535546153784], [-0.012349440716207027], [0.01729443110525608]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.08053532242774963], [0.003142738714814186], [0.08117058873176575], [-0.04404306039214134]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_2472f1a1d7d6a00c4af2e9b42598ab3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [-0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-13.607486724853516], [1.4584904909133911], [7.57281494140625], [3.546661615371704]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_78a96c00e7d75950e6e509e9740cb87f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65c88b40e6c2c056df9e04e1ee0eaaab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2031, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2031, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_353e74d6f2a325cc6dd4bcf3fe7eb793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_353e74d6f2a325cc6dd4bcf3fe7eb793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_353e74d6f2a325cc6dd4bcf3fe7eb793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_353e74d6f2a325cc6dd4bcf3fe7eb793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_353e74d6f2a325cc6dd4bcf3fe7eb793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_353e74d6f2a325cc6dd4bcf3fe7eb793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_353e74d6f2a325cc6dd4bcf3fe7eb793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_353e74d6f2a325cc6dd4bcf3fe7eb793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_353e74d6f2a325cc6dd4bcf3fe7eb793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_353e74d6f2a325cc6dd4bcf3fe7eb793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_353e74d6f2a325cc6dd4bcf3fe7eb793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2031, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_65c88b40e6c2c056df9e04e1ee0eaaab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2031, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2031, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_e84eacfc7586799793b1b18333dc22cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.008263586089015007, 0.26963362097740173, 0.15679161250591278, 0.48167404532432556], [0.29430466890335083, 0.29705536365509033, 0.25334876775741577, 0.022261887788772583], [0.2445625215768814, 0.38511574268341064, 0.39862141013145447, 0.056395020335912704], [0.2445625215768814, 0.38511574268341064, 0.39862141013145447, 0.056395020335912704], [0.11120018362998962, 0.08992772549390793, 0.25522947311401367, 0.10254336893558502]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.03961222618818283, 0.3033975064754486, 0.0521332286298275, 0.4385668933391571], [0.02600282058119774, 0.010099442675709724, 0.1543867290019989, 0.3911326825618744], [0.022051092237234116, 0.031336959451436996, 0.2186094969511032, 0.31589949131011963], [0.022051092237234116, 0.031336959451436996, 0.2186094969511032, 0.31589949131011963], [0.25734516978263855, 0.3669837713241577, 0.0910768210887909, 0.41226935386657715]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_84f53858cc5adf24da52014a16a4d24f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([4205, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4205, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99ac1322e8a02d7369c36cf856f7f81e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99ac1322e8a02d7369c36cf856f7f81e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99ac1322e8a02d7369c36cf856f7f81e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99ac1322e8a02d7369c36cf856f7f81e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99ac1322e8a02d7369c36cf856f7f81e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99ac1322e8a02d7369c36cf856f7f81e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99ac1322e8a02d7369c36cf856f7f81e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99ac1322e8a02d7369c36cf856f7f81e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99ac1322e8a02d7369c36cf856f7f81e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99ac1322e8a02d7369c36cf856f7f81e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99ac1322e8a02d7369c36cf856f7f81e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4205, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_84f53858cc5adf24da52014a16a4d24f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([4205, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4205, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_137706462056ed3e627aa50c2b218ff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12940450012683868, 0.20436866581439972, 0.030623840168118477, 0.09573113918304443], [0.19299115240573883, 0.05547969043254852, 0.020191796123981476, 0.0483412966132164], [0.0833970308303833, 0.3081290125846863, 0.025063782930374146, 0.38557013869285583], [0.12940450012683868, 0.20436866581439972, 0.030623840168118477, 0.09573113918304443], [0.43072736263275146, 0.33830726146698, 0.034355126321315765, 0.45046713948249817], [0.15332646667957306, 0.31060591340065, 0.4668656885623932, 0.10964678227901459], [0.43072736263275146, 0.33830726146698, 0.034355126321315765, 0.45046713948249817]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.4300585985183716, 0.3968757390975952, 0.21152296662330627, 0.15867376327514648], [0.2711958885192871, 0.3687015473842621, 0.361213743686676, 0.4051762819290161], [0.4076938033103943, 0.39888378977775574, 0.12745285034179688, 0.3103872537612915], [0.4300585985183716, 0.3968757390975952, 0.21152296662330627, 0.15867376327514648], [0.408200204372406, 0.48546579480171204, 0.22410252690315247, 0.04801633208990097], [0.013495078310370445, 0.37293046712875366, 0.13591356575489044, 0.21783286333084106], [0.408200204372406, 0.48546579480171204, 0.22410252690315247, 0.04801633208990097]], dtype='float32').reshape([7, 4]),
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