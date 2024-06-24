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


class TestPrimitiveOp_d0373638288ad1ab7edfb96d2f6d908a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.34312447905540466]], [[0.29233741760253906]], [[0.13005630671977997]], [[0.11632351577281952]], [[0.3264560103416443]], [[0.2895161211490631]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.5729690194129944]], [[0.5418186187744141]], [[0.7066535949707031]], [[0.5188155174255371]], [[0.7043853402137756]], [[0.6182553768157959]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_5b65c5fbfeb7d2ca2b6767307ade46fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.015447561629116535]], [[0.03673126921057701]], [[0.23140306770801544]], [[0.38015881180763245]], [[0.2941504120826721]], [[0.18359073996543884]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.5874362587928772]], [[0.8193960785865784]], [[0.6515163779258728]], [[0.5629575252532959]], [[0.5636677145957947]], [[0.5595793128013611]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_e0aa74f3797869ea1de1f7af4de3a7f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.19983968138694763, 0.23413076996803284]], [[0.4779529869556427, 0.26637715101242065]], [[0.008956856094300747, 0.13959631323814392]], [[0.25195440649986267, 0.2123015969991684]], [[0.07262172549962997, 0.003288618288934231]], [[0.0338602252304554, 0.13654693961143494]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.08827981352806091, 0.42714759707450867]], [[0.40179383754730225, 0.16039198637008667]], [[0.4927435517311096, 0.21667078137397766]], [[0.43478116393089294, 0.07529047876596451]], [[0.4230801463127136, 0.47436094284057617]], [[0.13252036273479462, 0.3180251717567444]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_ce9abdb3993c2143a7569bcfb29e1ddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.1398528814315796, 0.17465122044086456]], [[0.3099498450756073, 0.15525367856025696]], [[0.11630041152238846, 0.32290175557136536]], [[0.14881692826747894, 0.313494473695755]], [[0.34778374433517456, 0.07256877422332764]], [[0.472201943397522, 0.0268948245793581]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.08827981352806091, 0.42714759707450867]], [[0.40179383754730225, 0.16039198637008667]], [[0.4927435517311096, 0.21667078137397766]], [[0.43478116393089294, 0.07529047876596451]], [[0.4230801463127136, 0.47436094284057617]], [[0.13252036273479462, 0.3180251717567444]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class TestPrimitiveOp_a7377ee47a9000718a5ba5ad2f8b9dad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b0071a67648dd3072ffe70503722fe9
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4153757393360138, 0.4384436011314392]], [[0.47887399792671204, 0.23715141415596008]], [[0.15636569261550903, 0.2781256139278412]], [[0.23899498581886292, 0.34275150299072266]], [[0.23657719790935516, 0.4904538094997406]], [[0.39618203043937683, 0.2609063982963562]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class TestPrimitiveOp_5eeda49fdd22ac4beedd22ddb098ed6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
            paddle.to_tensor([0.14145363867282867, 0.17830027639865875, 0.19601845741271973, 0.14128723740577698, 0.1610289216041565, 0.0832785964012146, 0.2927471101284027, 0.44636663794517517, 0.25815659761428833, 0.017593204975128174, 0.030450286343693733, 0.4521762430667877, 0.14131401479244232, 0.2590143084526062, 0.4330943524837494, 0.2948601245880127], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_5759404ba19869ad9fc649ca8106672b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14145363867282867, 0.17830027639865875, 0.19601845741271973, 0.14128723740577698, 0.1610289216041565, 0.0832785964012146, 0.2927471101284027, 0.44636663794517517, 0.25815659761428833, 0.017593204975128174, 0.030450286343693733, 0.4521762430667877, 0.14131401479244232, 0.2590143084526062, 0.4330943524837494, 0.2948601245880127], dtype='float32').reshape([16]),
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


class TestPrimitiveOp_97d2f6cf87134922040a7a2f79ba820a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1841, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_0e835a7194d2019c20d40a10533521e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e835a7194d2019c20d40a10533521e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e835a7194d2019c20d40a10533521e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e835a7194d2019c20d40a10533521e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e835a7194d2019c20d40a10533521e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e835a7194d2019c20d40a10533521e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e835a7194d2019c20d40a10533521e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e835a7194d2019c20d40a10533521e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e835a7194d2019c20d40a10533521e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e835a7194d2019c20d40a10533521e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e835a7194d2019c20d40a10533521e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_97d2f6cf87134922040a7a2f79ba820a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1841, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_48dda26e9ebc73410a34103c85370c11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12660089135169983, 0.4498346149921417, 0.16728460788726807, 0.19079479575157166], [0.48984792828559875, 0.21813873946666718, 0.05808158963918686, 0.4661862552165985], [0.15069082379341125, 0.14072011411190033, 0.10790818929672241, 0.0820726752281189], [0.20878872275352478, 0.4584120213985443, 0.39713698625564575, 0.19254882633686066], [0.44483596086502075, 0.496868759393692, 0.4679279625415802, 0.15570507943630219]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.19260439276695251, 0.326575368642807, 0.02678137645125389, 0.28417864441871643], [0.23691777884960175, 0.022842297330498695, 0.09383618831634521, 0.0929613783955574], [0.09306998550891876, 0.471201092004776, 0.4177243709564209, 0.12903742492198944], [0.3448992967605591, 0.060399431735277176, 0.3199009597301483, 0.08565649390220642], [0.22654138505458832, 0.23738156259059906, 0.27291521430015564, 0.07365244626998901]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_96235eef8d672bd8102d7b54a01b7a85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.347147673368454, 0.3085428476333618, 0.4240269660949707, 0.246980682015419], [0.21620531380176544, 0.10374244302511215, 0.44789963960647583, 0.362010657787323], [0.15193185210227966, 0.19361913204193115, 0.1682417094707489, 0.08387500047683716], [0.21620531380176544, 0.10374244302511215, 0.44789963960647583, 0.362010657787323], [0.15193185210227966, 0.19361913204193115, 0.1682417094707489, 0.08387500047683716]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.05382548272609711, 0.48797520995140076, 0.42228272557258606, 0.13868021965026855], [0.2654235064983368, 0.3144491910934448, 0.07049560546875, 0.22292351722717285], [0.12629713118076324, 0.02664477936923504, 0.1750180870294571, 0.20296752452850342], [0.2654235064983368, 0.3144491910934448, 0.07049560546875, 0.22292351722717285], [0.12629713118076324, 0.02664477936923504, 0.1750180870294571, 0.20296752452850342]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_51a3ff92b364618dbfc941c7a753317a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22089730203151703], [0.007254623807966709], [0.08551525324583054], [0.28749170899391174], [0.03921782970428467], [0.38058915734291077], [0.00696443160995841], [0.09035753458738327], [0.016163386404514313]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.2136506289243698], [0.44700920581817627], [0.4920419454574585], [0.324871301651001], [0.4347423315048218], [0.11308283358812332], [0.49398526549339294], [0.2829873263835907], [0.371082603931427]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_af2c6219c9235b84ae1c88619fd2887c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.032230887562036514], [0.18736937642097473], [0.015003582462668419], [0.03740933537483215], [0.09220337867736816], [0.21914060413837433], [0.005615626461803913], [0.10680714249610901], [0.10162022709846497]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.4841810464859009], [0.32271942496299744], [0.44343236088752747], [0.38031214475631714], [0.2671041786670685], [0.46841099858283997], [0.4129534661769867], [0.2900417149066925], [0.2420433908700943]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_f2ea3f85925b805a013c2e11f7745124(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.48502784967422485], [0.007254623807966709], [0.08551525324583054], [0.28749170899391174], [0.42672285437583923], [0.4068581461906433], [0.00696443160995841], [0.11411301791667938], [0.016163386404514313]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.2136506289243698], [0.44700920581817627], [0.4920419454574585], [0.324871301651001], [0.4347423315048218], [0.11308283358812332], [0.4580673575401306], [0.2829873263835907], [0.09454172104597092]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_d3c8095f439bcd2426aefbe821a4d511(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.43891218304634094], [0.3814011812210083], [0.4228847026824951], [0.03740933537483215], [0.2268114984035492], [0.26206380128860474], [0.4565003216266632], [0.3727211654186249], [0.10162022709846497]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3358563482761383], [0.32271942496299744], [0.44343236088752747], [7.760437438264489e-05], [0.2671041786670685], [0.39811384677886963], [0.4129534661769867], [0.08367976546287537], [0.2420433908700943]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_0d69e879fd6e6c4200a51c9974ff4f31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22089730203151703], [0.033434219658374786], [0.16276678442955017], [0.3074219524860382], [0.03921782970428467], [0.38058915734291077], [0.19385917484760284], [0.09035753458738327], [0.05179273337125778]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.05196747928857803], [0.3380136489868164], [0.4891405999660492], [0.2964899241924286], [0.20674164593219757], [0.010906912386417389], [0.49398526549339294], [0.1478980928659439], [0.371082603931427]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_66c11986925b579e755ed2a11f65310c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.032230887562036514], [0.18736937642097473], [0.015003582462668419], [0.33160749077796936], [0.09220337867736816], [0.21914060413837433], [0.005615626461803913], [0.10680714249610901], [0.24876168370246887]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.4841810464859009], [0.08505020290613174], [0.18903541564941406], [0.38031214475631714], [0.10913760215044022], [0.46841099858283997], [0.29409098625183105], [0.2900417149066925], [0.05474267899990082]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_a55d6560abbb8ef73f58dc6d174a7519(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.04838085174560547], [-0.056969888508319855], [0.065152607858181], [-0.0019278854597359896], [0.0031600119546055794], [-0.13211898505687714], [0.06693486869335175], [-0.03826824575662613], [-0.05094216763973236]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_9c1d2b3ae464b4c3a155a4462f68f65a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.48502784967422485], [0.033434219658374786], [0.16276678442955017], [0.3074219524860382], [0.42672285437583923], [0.4068581461906433], [0.19385917484760284], [0.11411301791667938], [0.05179273337125778]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.05196747928857803], [0.3380136489868164], [0.4891405999660492], [0.2964899241924286], [0.20674164593219757], [0.010906912386417389], [0.4580673575401306], [0.1478980928659439], [0.09454172104597092]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_6311e550d5f4d661f776a4cada7c16db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.43891218304634094], [0.3814011812210083], [0.4228847026824951], [0.33160749077796936], [0.2268114984035492], [0.26206380128860474], [0.4565003216266632], [0.3727211654186249], [0.24876168370246887]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3358563482761383], [0.08505020290613174], [0.18903541564941406], [7.760437438264489e-05], [0.10913760215044022], [0.39811384677886963], [0.29409098625183105], [0.08367976546287537], [0.05474267899990082]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_323cee221131b3696a732f76a0815e94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.04462939873337746], [-0.09026241302490234], [-0.0763222873210907], [0.0036242941860109568], [0.025886045768857002], [-0.053869184106588364], [-0.042909879237413406], [-0.009765285067260265], [-0.008294115774333477]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.04838085174560547], [-0.056969888508319855], [0.065152607858181], [-0.0019278853433206677], [0.0031600119546055794], [-0.13211898505687714], [0.06693486869335175], [-0.03826824575662613], [-0.05094216763973236]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_9d73168b0bf0ba10da58b25c439e4595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [-0.0], [0.0], [-0.0], [0.0], [-0.0], [0.0], [-0.0], [-0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[2.0840578079223633], [0.3688414990901947], [1.8536512851715088], [1.5319340229034424], [0.8779259920120239], [-1.4525892734527588], [2.55989408493042], [-2.91880464553833], [-5.141964912414551]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_b683a7538b123f277ffef55f48129604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_495e93b818db93a655a04d3e9cac643c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.46325910091400146]], [[0.2570236921310425]], [[0.2363130748271942]], [[0.05351507291197777]], [[0.15206843614578247]], [[0.1988680213689804]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6231030821800232]], [[0.6150645017623901]], [[0.6048080325126648]], [[0.7294567227363586]], [[0.500393807888031]], [[0.7052760720252991]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_09fb7b24a8d1178f43a13a4d81a25ce9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab26114d7983355cd961d75de1b302b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3885578513145447]], [[0.3919214606285095]], [[0.4855879545211792]], [[0.2247704416513443]], [[0.3383931815624237]], [[0.37155577540397644]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6473247408866882]], [[0.6579882502555847]], [[0.7751756906509399]], [[0.5198081731796265]], [[0.517557680606842]], [[0.7182232737541199]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_fe8db4469b36c12b3638ae5d33d0bf5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([5562, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57b8196f950ebd14ae5d23cd0c6daa4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57b8196f950ebd14ae5d23cd0c6daa4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57b8196f950ebd14ae5d23cd0c6daa4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57b8196f950ebd14ae5d23cd0c6daa4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57b8196f950ebd14ae5d23cd0c6daa4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57b8196f950ebd14ae5d23cd0c6daa4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57b8196f950ebd14ae5d23cd0c6daa4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57b8196f950ebd14ae5d23cd0c6daa4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57b8196f950ebd14ae5d23cd0c6daa4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57b8196f950ebd14ae5d23cd0c6daa4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57b8196f950ebd14ae5d23cd0c6daa4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_fe8db4469b36c12b3638ae5d33d0bf5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([5562, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dbdec6787c315bd73013baf293935157(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.008119892328977585, 0.44394004344940186, 0.33099478483200073, 0.3245946168899536], [0.12415602058172226, 0.001519524957984686, 0.02916640043258667, 0.41793692111968994], [0.3544696569442749, 0.16319260001182556, 0.3163454830646515, 0.420857310295105], [0.12415602058172226, 0.001519524957984686, 0.02916640043258667, 0.41793692111968994], [0.3544696569442749, 0.16319260001182556, 0.3163454830646515, 0.420857310295105], [0.16596762835979462, 0.10555141419172287, 0.4166881740093231, 0.22149455547332764], [0.16596762835979462, 0.10555141419172287, 0.4166881740093231, 0.22149455547332764]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.43433088064193726, 0.31841519474983215, 0.1970967948436737, 0.07437502592802048], [0.026410501450300217, 0.29843226075172424, 0.3641354441642761, 0.020145010203123093], [0.2909087836742401, 0.14358781278133392, 0.22863580286502838, 0.03550170361995697], [0.026410501450300217, 0.29843226075172424, 0.3641354441642761, 0.020145010203123093], [0.2909087836742401, 0.14358781278133392, 0.22863580286502838, 0.03550170361995697], [0.27456945180892944, 0.2608335018157959, 0.08822476118803024, 0.27658140659332275], [0.27456945180892944, 0.2608335018157959, 0.08822476118803024, 0.27658140659332275]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_e8f1cb014144abb4def8cf6291a7ff3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0693720355629921, 0.37168705463409424, 0.3194563388824463, 0.31869107484817505, 0.14553475379943848, 0.33666518330574036], dtype='float32').reshape([6]),
            paddle.to_tensor([0.14741972088813782, 0.1336534470319748, 0.11528350412845612, 0.4567588269710541, 0.24708279967308044, 0.3292865753173828], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ca14a27c708362986f4ae2b2fa62fb1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3068593740463257, 0.08607549220323563, 0.11235673725605011, 0.14383947849273682, 0.45763009786605835, 0.3360748291015625], dtype='float32').reshape([6]),
            paddle.to_tensor([0.25797852873802185, 0.12108806520700455, 0.2023414820432663, 0.36508259177207947, 0.0434037484228611, 0.07065112143754959], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_0cf094604ccf30beab9a467dba80f528(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22594380378723145, 0.10565175116062164, 0.03872787579894066, 0.2721693217754364, 0.14351142942905426, 0.3796778917312622], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2984810769557953, 0.018444553017616272, 0.06930149346590042, 0.3149198293685913, 0.005490172654390335, 0.3335161805152893], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_60d1a540916680b3c870a665357b5599(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1421709507703781, 0.1708022505044937, 0.1709064394235611, 0.046536415815353394, 0.06752748787403107, 0.041682589799165726], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2929052710533142, 0.40369516611099243, 0.1233789473772049, 0.3182181119918823, 0.267822802066803, 0.4557313323020935], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_b0064e1dbb1eb5a6274084b70f2b394d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14741972088813782, 0.10565175116062164, 0.03872787579894066, 0.2721693217754364, 0.14351142942905426, 0.33666518330574036], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2984810769557953, 0.1336534470319748, 0.11528350412845612, 0.4567588269710541, 0.24708279967308044, 0.3335161805152893], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_e0d64f4ac20e3d10b58720632a3a64ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1421709507703781, 0.12108806520700455, 0.1709064394235611, 0.046536415815353394, 0.06752748787403107, 0.041682589799165726], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2929052710533142, 0.40369516611099243, 0.2023414820432663, 0.36508259177207947, 0.267822802066803, 0.4557313323020935], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_584688f4f99a343437cdeee793acd061(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14741972088813782, 0.37168705463409424, 0.3194563388824463, 0.4567588269710541, 0.24708279967308044, 0.33666518330574036], dtype='float32').reshape([6]),
            paddle.to_tensor([0.14741972088813782, 0.1336534470319748, 0.11528350412845612, 0.4567588269710541, 0.24708279967308044, 0.3292865753173828], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_3c6198e6faf92f5581e5495e9a420614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3068593740463257, 0.12108806520700455, 0.2023414820432663, 0.36508259177207947, 0.45763009786605835, 0.3360748291015625], dtype='float32').reshape([6]),
            paddle.to_tensor([0.25797852873802185, 0.12108806520700455, 0.2023414820432663, 0.36508259177207947, 0.0434037484228611, 0.07065112143754959], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_1ad802f369fe294fd7d8637d9491f907(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.01093385647982359, -0.020309938117861748, -0.0014530873158946633, 0.011614530347287655, -0.027645012363791466, -0.017154740169644356], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, -0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_c11e0e4d3f3730d30d5b91784a056a4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10839587450027466, 0.2526702582836151, 0.2173699140548706, 0.38772493600845337, 0.19630877673625946, 0.3329758644104004], dtype='float32').reshape([6]),
            paddle.to_tensor([0.26221245527267456, 0.06204815208911896, 0.05401468276977539, 0.29354459047317505, 0.07450079917907715, 0.35659703612327576], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_42c6139f484b322a4f3cfbd6d52f9243(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.28241896629333496, 0.10358177870512009, 0.1573491096496582, 0.25446105003356934, 0.2505169212818146, 0.20336297154426575], dtype='float32').reshape([6]),
            paddle.to_tensor([0.21753811836242676, 0.2872487008571625, 0.147142693400383, 0.18237726390361786, 0.16767513751983643, 0.24870696663856506], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_6e9a2a4e4ad0a4f5bf0ef88fff61b862(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22594380378723145, 0.37168705463409424, 0.3194563388824463, 0.4567588269710541, 0.24708279967308044, 0.3796778917312622], dtype='float32').reshape([6]),
            paddle.to_tensor([0.14741972088813782, 0.018444553017616272, 0.06930149346590042, 0.3149198293685913, 0.005490172654390335, 0.3292865753173828], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_7bef68378c736db9de73f3237fcc9f08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3068593740463257, 0.1708022505044937, 0.2023414820432663, 0.36508259177207947, 0.45763009786605835, 0.3360748291015625], dtype='float32').reshape([6]),
            paddle.to_tensor([0.25797852873802185, 0.12108806520700455, 0.1233789473772049, 0.3182181119918823, 0.0434037484228611, 0.07065112143754959], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_b3f65bef674439de285f7e707eb36478(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.448515921831131, -0.35829007625579834, -0.5716385841369629, 0.1560753583908081, -0.603365421295166, -0.11103008687496185], dtype='float32').reshape([6]),
            paddle.to_tensor([-1.0112667083740234, -1.4247527122497559, -1.15567946434021, 0.5579191446304321, -0.24040982127189636, 0.027792196720838547], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_5c9c28eb2e3445db9fb50bf4681ab8ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1734, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5abce1a39cc00a48cb1861dd6e5b5ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5abce1a39cc00a48cb1861dd6e5b5ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5abce1a39cc00a48cb1861dd6e5b5ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5abce1a39cc00a48cb1861dd6e5b5ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5abce1a39cc00a48cb1861dd6e5b5ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5abce1a39cc00a48cb1861dd6e5b5ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5abce1a39cc00a48cb1861dd6e5b5ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5abce1a39cc00a48cb1861dd6e5b5ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5abce1a39cc00a48cb1861dd6e5b5ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5abce1a39cc00a48cb1861dd6e5b5ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5abce1a39cc00a48cb1861dd6e5b5ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_5c9c28eb2e3445db9fb50bf4681ab8ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1734, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_14f876d8d465fbe6a53409eb9712dce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
            paddle.to_tensor([0.10497827082872391, 0.24310307204723358, 0.481964647769928, 0.0501120500266552, 0.17494754493236542, 0.34785979986190796, 0.41855791211128235, 0.3518923223018646, 0.05826127529144287, 0.19665104150772095, 0.3200782537460327, 0.37988394498825073, 0.2087623029947281, 0.45571592450141907, 0.08767567574977875, 0.0012626966927200556, 0.2798956632614136, 0.43433037400245667, 0.4705473780632019, 0.01220680121332407, 0.16760872304439545, 0.1691567450761795, 0.4557212293148041, 0.361501008272171], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_d552087bddfbdf8ed04dff9b20d1dbff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10497827082872391, 0.24310307204723358, 0.481964647769928, 0.0501120500266552, 0.17494754493236542, 0.34785979986190796, 0.41855791211128235, 0.3518923223018646, 0.05826127529144287, 0.19665104150772095, 0.3200782537460327, 0.37988394498825073, 0.2087623029947281, 0.45571592450141907, 0.08767567574977875, 0.0012626966927200556, 0.2798956632614136, 0.43433037400245667, 0.4705473780632019, 0.01220680121332407, 0.16760872304439545, 0.1691567450761795, 0.4557212293148041, 0.361501008272171], dtype='float32').reshape([24]),
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


class TestPrimitiveOp_6c8e8ec1c29be89e42c4c20a93819ef0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1541, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f194235b1b7fb9ff9779378d1404badd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f194235b1b7fb9ff9779378d1404badd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f194235b1b7fb9ff9779378d1404badd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f194235b1b7fb9ff9779378d1404badd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f194235b1b7fb9ff9779378d1404badd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f194235b1b7fb9ff9779378d1404badd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f194235b1b7fb9ff9779378d1404badd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f194235b1b7fb9ff9779378d1404badd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f194235b1b7fb9ff9779378d1404badd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f194235b1b7fb9ff9779378d1404badd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f194235b1b7fb9ff9779378d1404badd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_6c8e8ec1c29be89e42c4c20a93819ef0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1541, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_4cdbfad22d19067d11513aa38ca993ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
            paddle.to_tensor([0.20741339027881622, 0.24170714616775513, 0.16768890619277954, 0.05824856087565422], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_2bddbc5b2cf9a22cf6a6861e28a690ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20741339027881622, 0.24170714616775513, 0.16768890619277954, 0.05824856087565422], dtype='float32').reshape([4]),
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


class TestPrimitiveOp_3356b96cdb787f3a77b48ae90277ff56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3862595856189728, 0.2800934612751007, 0.3698129951953888, 0.03467046096920967], [0.07365134358406067, 0.4580913782119751, 0.1571710705757141, 0.23716704547405243], [0.17101317644119263, 0.2147289216518402, 0.08336327970027924, 0.23207329213619232], [0.45301198959350586, 0.24664044380187988, 0.013105637393891811, 0.4136313498020172], [0.45301198959350586, 0.24664044380187988, 0.013105637393891811, 0.4136313498020172], [0.17101317644119263, 0.2147289216518402, 0.08336327970027924, 0.23207329213619232]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.4210229814052582, 0.011279154568910599, 0.21361538767814636, 0.15670108795166016], [0.11573070287704468, 0.20195814967155457, 0.18034467101097107, 0.05424267798662186], [0.10406724363565445, 0.08381710201501846, 0.4485604465007782, 0.09263505786657333], [0.39866992831230164, 0.3768742084503174, 0.3094480633735657, 0.32597723603248596], [0.39866992831230164, 0.3768742084503174, 0.3094480633735657, 0.32597723603248596], [0.10406724363565445, 0.08381710201501846, 0.4485604465007782, 0.09263505786657333]], dtype='float32').reshape([6, 4]),
        ]


class TestPrimitiveOp_544fa61109446b890e71ec2d9bcbf203(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2663381099700928, 0.4811933636665344, 0.2132277935743332, 0.02409437857568264], [0.072791188955307, 0.26343002915382385, 0.45340073108673096, 0.1247192770242691], [0.028602590784430504, 0.1064571738243103, 0.35993966460227966, 0.21440844237804413], [0.03743935003876686, 0.48484158515930176, 0.3492962718009949, 0.16122804582118988], [0.2663381099700928, 0.4811933636665344, 0.2132277935743332, 0.02409437857568264]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.09686175733804703, 0.3512248992919922, 0.1370760202407837, 0.08551502972841263], [0.08279651403427124, 0.252678245306015, 0.2928794324398041, 0.33451133966445923], [0.3659399747848511, 0.12387955188751221, 0.3836943507194519, 0.2465433031320572], [0.38178277015686035, 0.08002657443284988, 0.39748862385749817, 0.2772524654865265], [0.09686175733804703, 0.3512248992919922, 0.1370760202407837, 0.08551502972841263]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_dc38fbca4d28435e70b9ea4f76b0680d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ed83e162b546319992eb260442636c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17654746770858765]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.46222788095474243]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_4503947cfa2f758980a974cc5a161203(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11643274873495102]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.25180718302726746]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_0b87286f2435bd6fcb93f11a13088537(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.30370238423347473]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.26397430896759033]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_f8d0fc5e7bd1e70e079b2de00b1310f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.34866175055503845]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.25180718302726746]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_5ed83e162b546319992eb260442636c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17654746770858765]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.46222788095474243]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_8835f8260811d14492453b0cee231192(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11643274873495102]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.13167576491832733]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_d91e7758d1dd67d0ed502f2885a1b8fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.008202476426959038]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_0b87286f2435bd6fcb93f11a13088537(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.30370238423347473]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.26397430896759033]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_7528a18ba34cb76208228a826b662554(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.34866175055503845]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.13167576491832733]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_62186d3281636ca9b9ef5e24924746ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.008620435371994972]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.008202476426959038]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_cb1888cb18fb840e5eb5d73a017e4bda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.048484668135643005]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_7730e578ab029c50dc4741a8c1681200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3089545667171478], [0.14597834646701813], [0.19613602757453918], [0.12908796966075897], [0.2358388751745224], [0.133039191365242]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3404685854911804], [0.3644312620162964], [0.44551825523376465], [0.496250182390213], [0.27458834648132324], [0.18564586341381073]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_3dc6fcf08494a0b634be7b0d01a26e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07759702950716019], [0.23825453221797943], [0.15273594856262207], [0.2640349566936493], [0.25769028067588806], [0.189493328332901]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.44728875160217285], [0.35537904500961304], [0.435450941324234], [0.4250192940235138], [0.3090977668762207], [0.04632904753088951]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_3e4197706399b1e2fd9bbf848ac7b6ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.332610547542572], [0.14597834646701813], [0.19613602757453918], [0.12908796966075897], [0.26270008087158203], [0.133039191365242]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3404685854911804], [0.3644312620162964], [0.44551825523376465], [0.496250182390213], [0.2719564139842987], [0.07249618321657181]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_33866c11cf41376df932b24897958526(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.40432968735694885], [0.2534337639808655], [0.18960238993167877], [0.41409391164779663], [0.25769028067588806], [0.23563654720783234]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3674864172935486], [0.35537904500961304], [0.435450941324234], [0.12282329052686691], [0.16958539187908173], [0.04632904753088951]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_f46c3e7653e6288e1f40c7f769b5f214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3089545667171478], [0.24098819494247437], [0.3093312382698059], [0.2416381686925888], [0.2358388751745224], [0.31183764338493347]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.03989572077989578], [0.06823953241109848], [0.13498272001743317], [0.24980278313159943], [0.27458834648132324], [0.18564586341381073]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_927697de3e093cee49bab77820c7d541(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07759702950716019], [0.23825453221797943], [0.15273594856262207], [0.2640349566936493], [0.366540789604187], [0.189493328332901]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.44728875160217285], [0.1407179981470108], [0.16133081912994385], [0.4250192940235138], [0.3090977668762207], [0.028634952381253242]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_9541d2ee611460ea4e7b8c5048a01020(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.09975834935903549], [0.039119549095630646], [0.059811756014823914], [-0.10562919825315475], [-0.003041415009647608], [0.03176024928689003]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_d46cd1d2bbbb7a6d29644982de960ddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.332610547542572], [0.24098819494247437], [0.3093312382698059], [0.2416381686925888], [0.26270008087158203], [0.31183764338493347]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.03989572077989578], [0.06823953241109848], [0.13498272001743317], [0.24980278313159943], [0.2719564139842987], [0.07249618321657181]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_24d90f6e5590257372a78f8624e91445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.40432968735694885], [0.2534337639808655], [0.18960238993167877], [0.41409391164779663], [0.366540789604187], [0.23563654720783234]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3674864172935486], [0.1407179981470108], [0.16133081912994385], [0.12282329052686691], [0.16958539187908173], [0.028634952381253242]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_28e430dabe84926c39122fdc782db4e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.01078457199037075], [0.01947149634361267], [0.004929106682538986], [-0.00237811217084527], [-0.0018230846617370844], [0.049544066190719604]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.09975834935903549], [0.039119549095630646], [0.059811756014823914], [-0.10562919825315475], [-0.003041415009647608], [0.03176024928689003]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_b17d361b3f0c254bd866023311b6c42e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [0.0], [0.0], [-0.0], [-0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[10.25009822845459], [-1.009067416191101], [-11.134400367736816], [-43.4172477722168], [-0.6682796478271484], [0.3589494824409485]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_64a1edb2c78f429fb318d760254a1687(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17250435054302216, 0.16421373188495636, 0.12080372869968414, 0.019319217652082443], [0.0635187178850174, 0.27851688861846924, 0.49873074889183044, 0.3256407380104065], [0.012831653468310833, 0.1793956756591797, 0.15398578345775604, 0.2377549111843109], [0.005950002931058407, 0.22988350689411163, 0.15722626447677612, 0.465120404958725]], dtype='float32').reshape([4, 4]),
            paddle.to_tensor([[0.2933735251426697, 0.25002679228782654, 0.16654841601848602, 0.4008951783180237], [0.4741528630256653, 0.3018854558467865, 0.43713486194610596, 0.36549216508865356], [0.2250722199678421, 0.2301221340894699, 0.03457648307085037, 0.23894113302230835], [0.2145441621541977, 0.4487716555595398, 0.045593637973070145, 0.14948105812072754]], dtype='float32').reshape([4, 4]),
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


class TestPrimitiveOp_01a7c6d38355a13cc474b241697479a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2061, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cbd1d64e7fe38f3547525565f79ab46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cbd1d64e7fe38f3547525565f79ab46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cbd1d64e7fe38f3547525565f79ab46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cbd1d64e7fe38f3547525565f79ab46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cbd1d64e7fe38f3547525565f79ab46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cbd1d64e7fe38f3547525565f79ab46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cbd1d64e7fe38f3547525565f79ab46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cbd1d64e7fe38f3547525565f79ab46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cbd1d64e7fe38f3547525565f79ab46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cbd1d64e7fe38f3547525565f79ab46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cbd1d64e7fe38f3547525565f79ab46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_01a7c6d38355a13cc474b241697479a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2061, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fdb3a0da87b4f020490f441c680332ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39730748534202576, 0.2787032127380371, 0.06083207577466965, 0.26745307445526123], [0.39730748534202576, 0.2787032127380371, 0.06083207577466965, 0.26745307445526123], [0.3657877743244171, 0.2783140540122986, 0.16101914644241333, 0.17535001039505005], [0.10173793882131577, 0.055175233632326126, 0.16765107214450836, 0.4223579466342926], [0.28543269634246826, 0.05586106702685356, 0.009063363075256348, 0.24863779544830322], [0.4391043186187744, 0.24367976188659668, 0.3932710886001587, 0.2582899332046509], [0.11062343418598175, 0.12585708498954773, 0.16156162321567535, 0.02637382224202156]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.4652462899684906, 0.0990920215845108, 0.4750223457813263, 0.03316156938672066], [0.4652462899684906, 0.0990920215845108, 0.4750223457813263, 0.03316156938672066], [0.16285353899002075, 0.3208737373352051, 0.1941397786140442, 0.15768243372440338], [0.4929409623146057, 0.085178442299366, 0.26607877016067505, 0.07390499860048294], [0.4131534993648529, 0.24290165305137634, 0.40154242515563965, 0.03222496807575226], [0.07790180295705795, 0.33458539843559265, 0.3260810375213623, 0.022301791235804558], [0.09355875849723816, 0.2630428373813629, 0.39191973209381104, 0.043721310794353485]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_d684a2f90201dee201a4369209b385f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bef5883090936e75a8eeb2b71fb35691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bef5883090936e75a8eeb2b71fb35691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bef5883090936e75a8eeb2b71fb35691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bef5883090936e75a8eeb2b71fb35691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bef5883090936e75a8eeb2b71fb35691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bef5883090936e75a8eeb2b71fb35691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bef5883090936e75a8eeb2b71fb35691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bef5883090936e75a8eeb2b71fb35691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bef5883090936e75a8eeb2b71fb35691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bef5883090936e75a8eeb2b71fb35691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bef5883090936e75a8eeb2b71fb35691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_d684a2f90201dee201a4369209b385f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4948b4e20de41f11c0ed52df63c6e437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_179ee9f91184b89e29d633851a40a187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_179ee9f91184b89e29d633851a40a187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_179ee9f91184b89e29d633851a40a187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_179ee9f91184b89e29d633851a40a187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_179ee9f91184b89e29d633851a40a187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_179ee9f91184b89e29d633851a40a187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_179ee9f91184b89e29d633851a40a187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_179ee9f91184b89e29d633851a40a187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_179ee9f91184b89e29d633851a40a187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_179ee9f91184b89e29d633851a40a187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_179ee9f91184b89e29d633851a40a187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_4948b4e20de41f11c0ed52df63c6e437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c9460cd58abff38d75364249aa3e12c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f27da2341bb4ad434c068d66fe0858d7
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_347443321cb777d33f96c5eed6867b15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2836502194404602, 0.4584534764289856, 0.27266594767570496, 0.4684521555900574], [0.009043618105351925, 0.28829532861709595, 0.005589806009083986, 0.007036004215478897], [0.009043618105351925, 0.28829532861709595, 0.005589806009083986, 0.007036004215478897], [0.3244793713092804, 0.39598217606544495, 0.20310303568840027, 0.3196679949760437], [0.23585490882396698, 0.1693139672279358, 0.13215239346027374, 0.39778029918670654], [0.1994534134864807, 0.1810484677553177, 0.19916170835494995, 0.41262897849082947]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.2825106680393219, 0.10053551197052002, 0.082350954413414, 0.08945828676223755], [0.15285807847976685, 0.027824314311146736, 0.29108482599258423, 0.2828611433506012], [0.15285807847976685, 0.027824314311146736, 0.29108482599258423, 0.2828611433506012], [0.1311185210943222, 0.23266805708408356, 0.3512043058872223, 0.467157781124115], [0.1659943163394928, 0.25516682863235474, 0.10842426866292953, 0.11381299048662186], [0.17552968859672546, 0.24801243841648102, 0.4070735573768616, 0.04634609445929527]], dtype='float32').reshape([6, 4]),
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


class TestPrimitiveOp_70e4d911299e56d15555071b54848f96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47492f828e721b67b39f9215d060228c
    def get_inputs(self):
        return [
            paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.3612814247608185, 0.04735812172293663, 5.087856292724609, 0.9870674014091492], [0.3634069859981537, 1.8980544805526733, 6.19453763961792, 0.3491046726703644]], dtype='float32').reshape([2, 4]),
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


class TestPrimitiveOp_f86bba5c214f851afa9266ab6d0792a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ff0558d29e4c36c82c5fce402c13bdb
    def get_inputs(self):
        return [
            paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[3.6202144622802734, 0.08586125075817108, 1.1028764247894287, 0.42400193214416504], [0.4315283000469208, 0.6178131103515625, 0.4066254198551178, 1.7470496892929077]], dtype='float32').reshape([2, 4]),
        ]


class TestPrimitiveOp_01a8dcccb2640d0b210bf27e1d821467(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.00047272868687286973], [0.056612178683280945], [0.17818866670131683], [0.2440360188484192], [0.006049283314496279]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.1571986973285675], [0.11458301544189453], [0.34768521785736084], [0.31836310029029846], [0.16771680116653442]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_aafb29bd75514aad0f048fb9849cb4c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.049238573759794235], [0.14470472931861877], [0.00861063040792942], [0.17378324270248413], [0.29924002289772034]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.3675876557826996], [0.3236313760280609], [0.3462056517601013], [0.4551815986633301], [0.469809353351593]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_b5eae92942c539afd0d9764170f18118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4107484221458435], [0.2350795716047287], [0.32436296343803406], [0.33859461545944214], [0.17683643102645874]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.1571986973285675], [0.11458301544189453], [0.34768521785736084], [0.20442333817481995], [0.04476496949791908]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_d1021a830576bc7be429e408a67545f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2710123062133789], [0.24374403059482574], [0.10803781449794769], [0.17378324270248413], [0.29924002289772034]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.3675876557826996], [0.2628155052661896], [0.3304205536842346], [0.4551815986633301], [0.469809353351593]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_70e61f913d7c97e7a563b6e16d4daa85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.00047272868687286973], [0.056612178683280945], [0.17818866670131683], [0.2440360188484192], [0.006049283314496279]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.008164001628756523], [0.1091982051730156], [0.34744080901145935], [0.31836310029029846], [0.16771680116653442]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_d2b0ebbab0fc88fbdb9f44155629398a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.049238573759794235], [0.14470472931861877], [0.00861063040792942], [0.40792182087898254], [0.3668667674064636]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.2649228572845459], [0.3236313760280609], [0.3462056517601013], [0.14429233968257904], [0.2508985102176666]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_dd8ad68b371f1250bd202fbcab91ec51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.022827766835689545], [0.007110994309186935], [0.06232514977455139], [-0.057350385934114456], [-0.041275642812252045]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_d310f681ccf288c757927858a5070e9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4107484221458435], [0.2350795716047287], [0.32436296343803406], [0.33859461545944214], [0.17683643102645874]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.008164001628756523], [0.1091982051730156], [0.34744080901145935], [0.20442333817481995], [0.04476496949791908]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_b1216de36a5652bc820e6d30022b626e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2710123062133789], [0.24374403059482574], [0.10803781449794769], [0.40792182087898254], [0.3668667674064636]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.2649228572845459], [0.2628155052661896], [0.3304205536842346], [0.14429233968257904], [0.2508985102176666]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_63ad41f2fd2070e3e5718f83334e8641(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0024515173863619566], [-0.00240074354223907], [0.005132114514708519], [0.035371504724025726], [0.015316097997128963]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.022827766835689545], [0.007110994309186935], [0.06232514977455139], [-0.057350385934114456], [-0.041275642812252045]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_4dfcf4ef7b0a7b5aec3b19b1f7d310a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [0.0], [0.0], [-0.0], [-0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[10.311688423156738], [3.9619967937469482], [-11.144145965576172], [2.621372699737549], [3.6949191093444824]], dtype='float32').reshape([5, 1]),
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


class TestPrimitiveOp_47d8bbf0b4b0a816697f3ce730566a57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2369, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba9329adf8b151b66e2eacb67e5bd0ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba9329adf8b151b66e2eacb67e5bd0ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba9329adf8b151b66e2eacb67e5bd0ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba9329adf8b151b66e2eacb67e5bd0ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba9329adf8b151b66e2eacb67e5bd0ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba9329adf8b151b66e2eacb67e5bd0ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba9329adf8b151b66e2eacb67e5bd0ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba9329adf8b151b66e2eacb67e5bd0ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba9329adf8b151b66e2eacb67e5bd0ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba9329adf8b151b66e2eacb67e5bd0ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba9329adf8b151b66e2eacb67e5bd0ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_47d8bbf0b4b0a816697f3ce730566a57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2369, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_751b6497eaf81c38fca8ad1656c28f68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([3054, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c96c0d946bbb6f918691640831a4783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c96c0d946bbb6f918691640831a4783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c96c0d946bbb6f918691640831a4783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c96c0d946bbb6f918691640831a4783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c96c0d946bbb6f918691640831a4783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c96c0d946bbb6f918691640831a4783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c96c0d946bbb6f918691640831a4783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c96c0d946bbb6f918691640831a4783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c96c0d946bbb6f918691640831a4783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c96c0d946bbb6f918691640831a4783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c96c0d946bbb6f918691640831a4783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_751b6497eaf81c38fca8ad1656c28f68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([3054, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90ce12f950be5aee6f348ef6381ee19f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([3819, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e041783ff5b1f5926fdd8f2a9a46a2ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e041783ff5b1f5926fdd8f2a9a46a2ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e041783ff5b1f5926fdd8f2a9a46a2ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e041783ff5b1f5926fdd8f2a9a46a2ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e041783ff5b1f5926fdd8f2a9a46a2ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e041783ff5b1f5926fdd8f2a9a46a2ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e041783ff5b1f5926fdd8f2a9a46a2ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e041783ff5b1f5926fdd8f2a9a46a2ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e041783ff5b1f5926fdd8f2a9a46a2ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e041783ff5b1f5926fdd8f2a9a46a2ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e041783ff5b1f5926fdd8f2a9a46a2ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_90ce12f950be5aee6f348ef6381ee19f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([3819, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_92c6a54366d58da3992f7258d4874508(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
            paddle.to_tensor([0.059932973235845566, 0.1609245091676712, 0.4488435387611389, 0.004289453383535147, 0.1617693454027176, 0.06451651453971863, 0.2686924338340759, 0.12058482319116592, 0.1321176439523697, 0.3679129481315613, 0.2524234354496002, 0.23021575808525085, 0.298456072807312, 0.45019611716270447, 0.49008193612098694, 0.017989452928304672, 0.4127422571182251, 0.24961227178573608, 0.19512465596199036, 0.470866858959198], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_1f8f1420ff7069b604cdac74741c87f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd8e39875659ab720f2be996b7ece3e4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.059932973235845566, 0.1609245091676712, 0.4488435387611389, 0.004289453383535147, 0.1617693454027176, 0.06451651453971863, 0.2686924338340759, 0.12058482319116592, 0.1321176439523697, 0.3679129481315613, 0.2524234354496002, 0.23021575808525085, 0.298456072807312, 0.45019611716270447, 0.49008193612098694, 0.017989452928304672, 0.4127422571182251, 0.24961227178573608, 0.19512465596199036, 0.470866858959198], dtype='float32').reshape([20]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_1840c0133b791f0d909e9d03b46a0763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.27045926451683044], [0.1253899484872818], [0.24463814496994019], [0.23376497626304626]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4831085503101349], [0.4750896692276001], [0.20687055587768555], [0.3251105844974518]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_774e5301b1a730ac5ab0c9393a6f238e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.04296104982495308], [0.3528343439102173], [0.2541756331920624], [0.22007668018341064]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.32444852590560913], [0.14037735760211945], [0.2893640995025635], [0.08842404931783676]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_897bb60cb6e0601feb40518a48dce1c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3838992714881897], [0.1253899484872818], [0.24463814496994019], [0.23376497626304626]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.06872285902500153], [0.06164400279521942], [0.20687055587768555], [0.3251105844974518]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_55a16f895f512ee81e805d54f2ca9e45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19007621705532074], [0.4756835699081421], [0.2541756331920624], [0.4143301248550415]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.32444852590560913], [0.14037735760211945], [0.08293063938617706], [0.08842404931783676]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_7210332732a9c65f293a7c3fefd77d5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.27045926451683044], [0.49347126483917236], [0.4834122955799103], [0.4336952865123749]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4831085503101349], [0.4750896692276001], [0.1859712302684784], [0.05235862359404564]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_765aac83f39624c397bec164ce8db333(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.04296104982495308], [0.3528343439102173], [0.31736236810684204], [0.22007668018341064]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.31226545572280884], [0.035297587513923645], [0.2893640995025635], [0.003905576653778553]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_77e5376750be626ed71eadebef26a07a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.014916401356458664], [0.027211245149374008], [0.014795346185564995], [0.052663881331682205]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_bada44e2cbd103003f98a098cc7e92fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3838992714881897], [0.49347126483917236], [0.4834122955799103], [0.4336952865123749]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.06872285902500153], [0.06164400279521942], [0.1859712302684784], [0.05235862359404564]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_33105c4a7e6d3c2e8fdd778c46fa405c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19007621705532074], [0.4756835699081421], [0.31736236810684204], [0.4143301248550415]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.31226545572280884], [0.035297587513923645], [0.08293063938617706], [0.003905576653778553]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_6751094000a264874bd5b7b5014607fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.03851116821169853], [0.19017067551612854], [0.06972962617874146], [0.15650993585586548]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.014916401356458664], [0.027211245149374008], [0.014795346185564995], [0.052663881331682205]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_a0dd61d359433b106fc6efb5a8706d66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[1.3873265981674194], [0.8569114208221436], [0.7878183126449585], [0.6635109782218933]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_78a96c00e7d75950e6e509e9740cb87f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e4dbb73fae59cad1ca0fba8b2c7a7a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2092, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_71f89436adb95ee62474ebb648657de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_71f89436adb95ee62474ebb648657de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_71f89436adb95ee62474ebb648657de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_71f89436adb95ee62474ebb648657de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_71f89436adb95ee62474ebb648657de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_71f89436adb95ee62474ebb648657de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_71f89436adb95ee62474ebb648657de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_71f89436adb95ee62474ebb648657de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_71f89436adb95ee62474ebb648657de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_71f89436adb95ee62474ebb648657de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_71f89436adb95ee62474ebb648657de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_7e4dbb73fae59cad1ca0fba8b2c7a7a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([2092, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_a98e59330acb73c80773d537808f4440(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4724057912826538, 0.005135802086442709, 0.35766884684562683, 0.02228466421365738], [0.2000947743654251, 0.15862610936164856, 0.3050123155117035, 0.37415891885757446], [0.20294500887393951, 0.11467795819044113, 0.23192065954208374, 0.3478229343891144], [0.20294500887393951, 0.11467795819044113, 0.23192065954208374, 0.3478229343891144], [0.21345028281211853, 0.2829083204269409, 0.19015470147132874, 0.31420353055000305]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.24325378239154816, 0.011176790110766888, 0.22048348188400269, 0.08810742199420929], [0.19158482551574707, 0.1266273558139801, 0.20404541492462158, 0.057321418076753616], [0.23941843211650848, 0.30055198073387146, 0.4333055317401886, 0.3577125668525696], [0.23941843211650848, 0.30055198073387146, 0.4333055317401886, 0.3577125668525696], [0.18810521066188812, 0.10880894213914871, 0.2213110774755478, 0.29051458835601807]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_a8850e6c0cb01ca571d2ae59c7d1fe56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([4214, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05d6d1eee501003f7c36cf0039d61aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05d6d1eee501003f7c36cf0039d61aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05d6d1eee501003f7c36cf0039d61aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05d6d1eee501003f7c36cf0039d61aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05d6d1eee501003f7c36cf0039d61aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05d6d1eee501003f7c36cf0039d61aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05d6d1eee501003f7c36cf0039d61aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05d6d1eee501003f7c36cf0039d61aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05d6d1eee501003f7c36cf0039d61aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05d6d1eee501003f7c36cf0039d61aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05d6d1eee501003f7c36cf0039d61aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dc2fd06667253249b7bc72958d8f0c0
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_a8850e6c0cb01ca571d2ae59c7d1fe56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b88a0911757050cc1660428654c44c3
    def get_inputs(self):
        return [
            paddle.uniform([4214, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d25fe089114764191e04447ecd06737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05d8aa4ed2eeb6cea990514729dfc697
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2305142879486084, 0.26621851325035095, 0.13248762488365173, 0.05205540731549263], [0.05531824007630348, 0.16881686449050903, 0.1733401119709015, 0.20085474848747253], [0.15340225398540497, 0.42899438738822937, 0.22799421846866608, 0.3197120428085327], [0.2305142879486084, 0.26621851325035095, 0.13248762488365173, 0.05205540731549263], [0.2244083285331726, 0.08819392323493958, 0.02766348607838154, 0.355491042137146], [0.45577722787857056, 0.14583490788936615, 0.2316286712884903, 0.17187027633190155], [0.2244083285331726, 0.08819392323493958, 0.02766348607838154, 0.355491042137146]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.4056689143180847, 0.04218864440917969, 0.19053590297698975, 0.14832772314548492], [0.46436965465545654, 0.34840816259384155, 0.012091579847037792, 0.16475367546081543], [0.12395384907722473, 0.35926690697669983, 0.02992437221109867, 0.03849229961633682], [0.4056689143180847, 0.04218864440917969, 0.19053590297698975, 0.14832772314548492], [0.10749533027410507, 0.4959408640861511, 0.1116555854678154, 0.37817081809043884], [0.49131840467453003, 0.16194528341293335, 0.0799216777086258, 0.3869289755821228], [0.10749533027410507, 0.4959408640861511, 0.1116555854678154, 0.37817081809043884]], dtype='float32').reshape([7, 4]),
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