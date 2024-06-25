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


class TestPrimitiveOp_6c8e8f9a1e3f19ae7563ce40598a1f53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dfcedf316eacbbadf7dd6cacaded896
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.32247185707092285]], [[0.1932452917098999]], [[0.38496288657188416]], [[0.132805734872818]], [[0.24269376695156097]], [[0.13803625106811523]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6919962763786316]], [[0.6690086722373962]], [[0.7589842081069946]], [[0.6850078105926514]], [[0.5845724940299988]], [[0.6995828151702881]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_56499b36a2e9c4389534648447077ed2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dfcedf316eacbbadf7dd6cacaded896
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.09298263490200043]], [[0.11903032660484314]], [[0.18748290836811066]], [[0.2656666040420532]], [[0.18859152495861053]], [[0.04117158055305481]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6984186172485352]], [[0.6834514141082764]], [[0.5561749935150146]], [[0.6638743877410889]], [[0.5413414239883423]], [[0.5201280117034912]]], dtype='float32').reshape([6, 1, 1]),
        ]


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


class TestPrimitiveOp_ae809a9a6d4e443bf018e305ea87e68b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae809a9a6d4e443bf018e305ea87e68b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae809a9a6d4e443bf018e305ea87e68b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae809a9a6d4e443bf018e305ea87e68b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae809a9a6d4e443bf018e305ea87e68b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae809a9a6d4e443bf018e305ea87e68b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae809a9a6d4e443bf018e305ea87e68b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_c060b8c79dd3faf4ef0837345eb5a79b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05a882ce170a29652e88a1d357dc5613
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_db8193251c597154b5bbf986edd68f79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db8193251c597154b5bbf986edd68f79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db8193251c597154b5bbf986edd68f79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db8193251c597154b5bbf986edd68f79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db8193251c597154b5bbf986edd68f79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db8193251c597154b5bbf986edd68f79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db8193251c597154b5bbf986edd68f79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_a63376dbd379cc3e9f1eccbf58e9eba5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3ded76ac568482376e2cdb80714a674
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_a3c22108d2b81c82c9440e88047b4d3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe63cab98da7d8138e0d1d86917b2864
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_693a35f419aeac89656bf62152a0072f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_693a35f419aeac89656bf62152a0072f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_693a35f419aeac89656bf62152a0072f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_693a35f419aeac89656bf62152a0072f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_693a35f419aeac89656bf62152a0072f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_693a35f419aeac89656bf62152a0072f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_693a35f419aeac89656bf62152a0072f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_078bd9ace561cea6e69653214a30bfa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_078bd9ace561cea6e69653214a30bfa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_078bd9ace561cea6e69653214a30bfa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_078bd9ace561cea6e69653214a30bfa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_078bd9ace561cea6e69653214a30bfa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_078bd9ace561cea6e69653214a30bfa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_078bd9ace561cea6e69653214a30bfa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2b4543a3e90da549ff8633afddbd98d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[8, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_009ed478ad9a700bc3fe38f28f267bf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b4543a3e90da549ff8633afddbd98d0
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_89393906ba88c86808ccb2c0774ed1db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2530a4e4625af688f4e3de9927836b64
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.23944492638111115, 0.1944647878408432]], [[0.44200199842453003, 0.09568645805120468]], [[0.38632965087890625, 0.12290971726179123]], [[0.36792171001434326, 0.2184819132089615]], [[0.49938255548477173, 0.31022676825523376]], [[0.45675018429756165, 0.0321899950504303]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.17827314138412476, 0.034787628799676895]], [[0.025618499144911766, 0.4306291937828064]], [[0.4350353181362152, 0.3191002607345581]], [[0.11267857998609543, 0.008304253220558167]], [[0.2059049755334854, 0.0397799089550972]], [[0.12073905020952225, 0.20204704999923706]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_fe6513786bad767778ffe18829e47339(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2530a4e4625af688f4e3de9927836b64
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.15506356954574585, 0.44753342866897583]], [[0.4900464117527008, 0.1239706426858902]], [[0.17805646359920502, 0.25923892855644226]], [[0.17756523191928864, 0.24519023299217224]], [[0.08149562031030655, 0.16336354613304138]], [[0.17411379516124725, 0.32042360305786133]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.17827314138412476, 0.034787628799676895]], [[0.025618499144911766, 0.4306291937828064]], [[0.4350353181362152, 0.3191002607345581]], [[0.11267857998609543, 0.008304253220558167]], [[0.2059049755334854, 0.0397799089550972]], [[0.12073905020952225, 0.20204704999923706]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class TestPrimitiveOp_1622bb128d4c0de5cfc1860f5e62d0c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38e60ad0c3265f689e713b6b70f444bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.10580560564994812, 0.049067698419094086]], [[0.22465749084949493, 0.35577359795570374]], [[0.48565107583999634, 0.1326971799135208]], [[0.08277522027492523, 0.2761727273464203]], [[0.05246429890394211, 0.4747205972671509]], [[0.1726287603378296, 0.28555411100387573]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class TestPrimitiveOp_bc1907066b2d16a9a9ec8d014f531d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc1907066b2d16a9a9ec8d014f531d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc1907066b2d16a9a9ec8d014f531d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc1907066b2d16a9a9ec8d014f531d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc1907066b2d16a9a9ec8d014f531d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc1907066b2d16a9a9ec8d014f531d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc1907066b2d16a9a9ec8d014f531d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_337405ec1fcc8883a54bd3ebe05ef83d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0368fad43623b73e790d949bddeb3e4a
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
            paddle.to_tensor([0.42532914876937866, 0.40336519479751587, 0.2862012982368469, 0.40241751074790955, 0.005774199031293392, 0.0750528946518898, 0.024343404918909073, 0.4737444221973419, 0.010925306007266045, 0.13228726387023926, 0.4787990152835846, 0.29970064759254456, 0.44392403960227966, 0.21227741241455078, 0.43791723251342773, 0.07732360810041428], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_2248d8d330128d2df3abeff98dcf880c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0368fad43623b73e790d949bddeb3e4a
    def get_inputs(self):
        return [
            paddle.to_tensor([0.42532914876937866, 0.40336519479751587, 0.2862012982368469, 0.40241751074790955, 0.005774199031293392, 0.0750528946518898, 0.024343404918909073, 0.4737444221973419, 0.010925306007266045, 0.13228726387023926, 0.4787990152835846, 0.29970064759254456, 0.44392403960227966, 0.21227741241455078, 0.43791723251342773, 0.07732360810041428], dtype='float32').reshape([16]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([16]),
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


class TestPrimitiveOp_f032e8640c782bd0a305029ba1823bf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f032e8640c782bd0a305029ba1823bf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f032e8640c782bd0a305029ba1823bf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f032e8640c782bd0a305029ba1823bf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f032e8640c782bd0a305029ba1823bf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f032e8640c782bd0a305029ba1823bf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f032e8640c782bd0a305029ba1823bf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
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


class TestPrimitiveOp_ae809a9a6d4e443bf018e305ea87e68b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae809a9a6d4e443bf018e305ea87e68b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae809a9a6d4e443bf018e305ea87e68b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae809a9a6d4e443bf018e305ea87e68b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae809a9a6d4e443bf018e305ea87e68b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae809a9a6d4e443bf018e305ea87e68b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae809a9a6d4e443bf018e305ea87e68b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9b4f7259eae77e25e8049906ea0a447
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_e5af25c119bbd17e3d9b11e40dc6757b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5af25c119bbd17e3d9b11e40dc6757b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5af25c119bbd17e3d9b11e40dc6757b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5af25c119bbd17e3d9b11e40dc6757b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5af25c119bbd17e3d9b11e40dc6757b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5af25c119bbd17e3d9b11e40dc6757b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5af25c119bbd17e3d9b11e40dc6757b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_a5ee4d058e977e28858d138f636591c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_098ae969f497f6d23855ee3a64346eab
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_db53f78b6790f8202abc17d46b7545d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db53f78b6790f8202abc17d46b7545d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db53f78b6790f8202abc17d46b7545d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db53f78b6790f8202abc17d46b7545d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db53f78b6790f8202abc17d46b7545d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db53f78b6790f8202abc17d46b7545d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db53f78b6790f8202abc17d46b7545d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_01813d371ac4f9e149eaa235d1f05bae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1758, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1758, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc5997dc79c2ca360499aa26c6ba1e84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01813d371ac4f9e149eaa235d1f05bae
    def get_inputs(self):
        return [
            paddle.uniform([1758, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6ac2341e58f9f1a762d472df0545789b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1758, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1758, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc2106e05b0633b6b87c6943756e6c57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ac2341e58f9f1a762d472df0545789b
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc2106e05b0633b6b87c6943756e6c57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ac2341e58f9f1a762d472df0545789b
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc2106e05b0633b6b87c6943756e6c57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ac2341e58f9f1a762d472df0545789b
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc2106e05b0633b6b87c6943756e6c57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ac2341e58f9f1a762d472df0545789b
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc2106e05b0633b6b87c6943756e6c57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ac2341e58f9f1a762d472df0545789b
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc2106e05b0633b6b87c6943756e6c57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ac2341e58f9f1a762d472df0545789b
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc2106e05b0633b6b87c6943756e6c57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ac2341e58f9f1a762d472df0545789b
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc2106e05b0633b6b87c6943756e6c57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ac2341e58f9f1a762d472df0545789b
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc2106e05b0633b6b87c6943756e6c57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ac2341e58f9f1a762d472df0545789b
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc2106e05b0633b6b87c6943756e6c57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ac2341e58f9f1a762d472df0545789b
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc2106e05b0633b6b87c6943756e6c57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ac2341e58f9f1a762d472df0545789b
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_e71b0a3d00d6453d20564b8fed96ebd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a4c83f203e9f67b723bc396bd4151be
    def get_inputs(self):
        return [
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_da5d7e2a679c607dfd94a865246289dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7e07e42ba4884aed6c7a3a1e4ff1f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc5997dc79c2ca360499aa26c6ba1e84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01813d371ac4f9e149eaa235d1f05bae
    def get_inputs(self):
        return [
            paddle.uniform([1758, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_012768c0db613e5ecd733a6a0e0edd3f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[5, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ae6dbf758bd9070a03a60d581d52f43b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_012768c0db613e5ecd733a6a0e0edd3f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10328175127506256, 0.21440859138965607, 0.20437446236610413, 0.3567073941230774], [0.44649216532707214, 0.4224661886692047, 0.19035965204238892, 0.1588238626718521], [0.04477078095078468, 0.3323558270931244, 0.2217559665441513, 0.25355425477027893], [0.30775442719459534, 0.012990595772862434, 0.4948490262031555, 0.4472769796848297], [0.1849198043346405, 0.4792657494544983, 0.4998842477798462, 0.05643796920776367]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.3600534498691559, 0.27924224734306335, 0.02028273418545723, 0.14265656471252441], [0.31524866819381714, 0.13639387488365173, 0.046474434435367584, 0.06295648217201233], [0.4942631423473358, 0.2419804334640503, 0.13529092073440552, 0.14985592663288116], [0.11564677953720093, 0.29416242241859436, 0.43282946944236755, 0.006130017340183258], [0.19085584580898285, 0.38278135657310486, 0.26800018548965454, 0.2910350561141968]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_96c1c418aa2419e5974f8b26da406b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96c1c418aa2419e5974f8b26da406b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96c1c418aa2419e5974f8b26da406b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96c1c418aa2419e5974f8b26da406b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96c1c418aa2419e5974f8b26da406b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96c1c418aa2419e5974f8b26da406b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96c1c418aa2419e5974f8b26da406b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_c0809e6f04bd84d889e2d6b72d4651ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0809e6f04bd84d889e2d6b72d4651ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0809e6f04bd84d889e2d6b72d4651ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0809e6f04bd84d889e2d6b72d4651ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0809e6f04bd84d889e2d6b72d4651ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0809e6f04bd84d889e2d6b72d4651ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0809e6f04bd84d889e2d6b72d4651ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_83e31c05be644baa3c0b4a3ad50d8725(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fabb9a4d545e157a51329090937d4782
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_306be0ac3ca44a4fa8aef692c0f0ca7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_012768c0db613e5ecd733a6a0e0edd3f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.008385724388062954, 0.3184845745563507, 0.21915414929389954, 0.2461262196302414], [0.1270468533039093, 0.08883246034383774, 0.38308197259902954, 0.3482043147087097], [0.0829109475016594, 0.44709712266921997, 0.41071271896362305, 0.40647557377815247], [0.1270468533039093, 0.08883246034383774, 0.38308197259902954, 0.3482043147087097], [0.0829109475016594, 0.44709712266921997, 0.41071271896362305, 0.40647557377815247]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.19000069797039032, 0.41332173347473145, 0.14485369622707367, 0.19007061421871185], [0.23585844039916992, 0.11523140966892242, 0.1360064595937729, 0.3488534986972809], [0.417148232460022, 0.12157418578863144, 0.39677926898002625, 0.48851677775382996], [0.23585844039916992, 0.11523140966892242, 0.1360064595937729, 0.3488534986972809], [0.417148232460022, 0.12157418578863144, 0.39677926898002625, 0.48851677775382996]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_1dc12a1fada6abd96fa33923fd8f8858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1dc12a1fada6abd96fa33923fd8f8858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1dc12a1fada6abd96fa33923fd8f8858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1dc12a1fada6abd96fa33923fd8f8858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1dc12a1fada6abd96fa33923fd8f8858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1dc12a1fada6abd96fa33923fd8f8858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1dc12a1fada6abd96fa33923fd8f8858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_9a5e07c698195cbd639733e10dbe90ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a5e07c698195cbd639733e10dbe90ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a5e07c698195cbd639733e10dbe90ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a5e07c698195cbd639733e10dbe90ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a5e07c698195cbd639733e10dbe90ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a5e07c698195cbd639733e10dbe90ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a5e07c698195cbd639733e10dbe90ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_2bca1c7cf336e07ea78fc3079879f5ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11550065129995346], [0.2585631310939789], [0.15645268559455872], [0.13411420583724976], [0.16898958384990692], [0.05408820882439613], [0.11748014390468597], [0.18900315463542938], [0.1729610115289688]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.4423205256462097], [0.4453306794166565], [0.2645948827266693], [0.3977745771408081], [0.25538399815559387], [0.4034620225429535], [0.2219296097755432], [0.3580722510814667], [0.44802966713905334]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_32acef3906f84f4dea0e6d76193bdd70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.044602248817682266], [0.11456053704023361], [0.3147315979003906], [0.03393561393022537], [0.21444787085056305], [0.029572542756795883], [0.15619488060474396], [0.0473758690059185], [0.3221006393432617]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.41259145736694336], [0.3572691082954407], [0.26499083638191223], [0.3949889540672302], [0.49846959114074707], [0.4767949879169464], [0.2055416703224182], [0.3756718039512634], [0.17685101926326752]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_0a8a7c7fde3069221ebf256588f543fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11550065129995346], [0.374041348695755], [0.15645268559455872], [0.2919510304927826], [0.16898958384990692], [0.05408820882439613], [0.11748014390468597], [0.40530943870544434], [0.1729610115289688]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.4423205256462097], [0.18592821061611176], [0.2645948827266693], [0.24683165550231934], [0.23438315093517303], [0.4034620225429535], [0.20104055106639862], [0.18984976410865784], [0.18868638575077057]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_17833b5f2e2eea0a615810a421fa4c47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.044602248817682266], [0.11456053704023361], [0.33949682116508484], [0.03393561393022537], [0.4222550392150879], [0.029572542756795883], [0.15619488060474396], [0.0473758690059185], [0.3221006393432617]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.24950331449508667], [0.34855425357818604], [0.26499083638191223], [0.0974547415971756], [0.49846959114074707], [0.30478185415267944], [0.2055416703224182], [0.3756718039512634], [0.030764833092689514]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_b1b283357576ff9a395f4c9e1d4ce1bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3067001700401306], [0.2585631310939789], [0.4714326858520508], [0.13411420583724976], [0.47448375821113586], [0.4214091897010803], [0.38982850313186646], [0.18900315463542938], [0.18645043671131134]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.12630529701709747], [0.4453306794166565], [0.027548898011446], [0.3977745771408081], [0.25538399815559387], [0.17480947077274323], [0.2219296097755432], [0.3580722510814667], [0.44802966713905334]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_99f77593e7a54174f64c36adfd2fea21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14979442954063416], [0.44975969195365906], [0.3147315979003906], [0.0915440171957016], [0.21444787085056305], [0.03469356149435043], [0.23302797973155975], [0.32175347208976746], [0.4466075897216797]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.41259145736694336], [0.3572691082954407], [0.1078881248831749], [0.3949889540672302], [0.296988844871521], [0.4767949879169464], [0.15632693469524384], [0.31878870725631714], [0.17685101926326752]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_654d9a762598404e992a76b26ea741f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.019558507949113846], [-0.06129153072834015], [0.08375721424818039], [0.0771404579281807], [-0.013100765645503998], [-0.012871161103248596], [0.017001457512378693], [-0.071235790848732], [-0.07514406740665436]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_a219fa753cf95b57289095f91496a0ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3067001700401306], [0.374041348695755], [0.4714326858520508], [0.2919510304927826], [0.47448375821113586], [0.4214091897010803], [0.38982850313186646], [0.40530943870544434], [0.18645043671131134]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.12630529701709747], [0.18592821061611176], [0.027548898011446], [0.24683165550231934], [0.23438315093517303], [0.17480947077274323], [0.20104055106639862], [0.18984976410865784], [0.18868638575077057]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_8dcd0940041b6a4f793547d949442d08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14979442954063416], [0.44975969195365906], [0.33949682116508484], [0.0915440171957016], [0.4222550392150879], [0.03469356149435043], [0.23302797973155975], [0.32175347208976746], [0.4466075897216797]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.24950331449508667], [0.34855425357818604], [0.1078881248831749], [0.0974547415971756], [0.296988844871521], [0.30478185415267944], [0.15632693469524384], [0.31878870725631714], [0.030764833092689514]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_5eea2036e9fca414e673454ae7b180ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.017986971884965897], [0.019038071855902672], [0.10280734300613403], [-0.000266688090050593], [0.030076488852500916], [-0.06660369783639908], [0.014480233192443848], [0.0006387874018400908], [-0.0009298031218349934]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.019558507949113846], [-0.06129153072834015], [0.08375721424818039], [0.0771404579281807], [-0.013100765645503998], [-0.012871161103248596], [0.017001457512378693], [-0.071235790848732], [-0.07514406740665436]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_2d04d23b06787b922faf8488890460de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [-0.0], [0.0], [0.0], [-0.0], [-0.0], [0.0], [-0.0], [-0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[2.0873708724975586], [4.219419002532959], [0.18529930710792542], [290.25347900390625], [1.4355815649032593], [0.8067500591278076], [-0.1741148978471756], [112.51721954345703], [-79.81718444824219]], dtype='float32').reshape([9, 1]),
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


class TestPrimitiveOp_bb960e02e1e87f655f64addd15635e49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_294b7063974dd0a73f2a3f686fb4d33d
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0b4aefc03ecfc6c3b50295eb9746013(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dfcedf316eacbbadf7dd6cacaded896
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.201649472117424]], [[0.3912675678730011]], [[0.2809157967567444]], [[0.3518129587173462]], [[0.296891987323761]], [[0.04682689532637596]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6271243095397949]], [[0.6748124361038208]], [[0.6373169422149658]], [[0.7895784378051758]], [[0.6225441098213196]], [[0.5134971141815186]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_b860e84c58e06d65d67373c1f5db97ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dfcedf316eacbbadf7dd6cacaded896
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.276872843503952]], [[0.2961027920246124]], [[0.05081995576620102]], [[0.41069820523262024]], [[0.07150100916624069]], [[0.018399052321910858]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6198098659515381]], [[0.7227193117141724]], [[0.7808607816696167]], [[0.5076951384544373]], [[0.5912706851959229]], [[0.5942469835281372]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_47c45ec036ab567b1b3c1f2e92cf00c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb93aefb6749274dd2abf6d9e2772012
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_afb65b2308d8802c7d65203b1972adff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5593, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[5593, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9cc5c3f5a7d886e836b34a4cba50cd18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afb65b2308d8802c7d65203b1972adff
    def get_inputs(self):
        return [
            paddle.uniform([5593, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_66dc6448c946967090f3d90991db803c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5593, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5593, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0134c59f4b31f2faa12dfdb76463f02c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66dc6448c946967090f3d90991db803c
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0134c59f4b31f2faa12dfdb76463f02c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66dc6448c946967090f3d90991db803c
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0134c59f4b31f2faa12dfdb76463f02c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66dc6448c946967090f3d90991db803c
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0134c59f4b31f2faa12dfdb76463f02c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66dc6448c946967090f3d90991db803c
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0134c59f4b31f2faa12dfdb76463f02c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66dc6448c946967090f3d90991db803c
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0134c59f4b31f2faa12dfdb76463f02c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66dc6448c946967090f3d90991db803c
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0134c59f4b31f2faa12dfdb76463f02c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66dc6448c946967090f3d90991db803c
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0134c59f4b31f2faa12dfdb76463f02c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66dc6448c946967090f3d90991db803c
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0134c59f4b31f2faa12dfdb76463f02c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66dc6448c946967090f3d90991db803c
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0134c59f4b31f2faa12dfdb76463f02c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66dc6448c946967090f3d90991db803c
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0134c59f4b31f2faa12dfdb76463f02c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66dc6448c946967090f3d90991db803c
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_856f39837953a2a04036dcea4980d314(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c7f912849971489ee33e3c279221b79
    def get_inputs(self):
        return [
            paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_cc7f31b525e4c3ddd4b60b1fef74f076(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5eb5f4b24b083f85bc38eeaaf2951db
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9cc5c3f5a7d886e836b34a4cba50cd18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afb65b2308d8802c7d65203b1972adff
    def get_inputs(self):
        return [
            paddle.uniform([5593, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a2f2b4663ad0e38d2e5e931474ae84ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[7, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_82561d7b0b3043a619d8946a5d8d1b8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2f2b4663ad0e38d2e5e931474ae84ac
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07080888003110886, 0.22228316962718964, 0.053138405084609985, 0.0774528905749321], [0.41907304525375366, 0.3542965352535248, 0.4363647997379303, 0.008938251063227654], [0.04087149351835251, 0.09524080157279968, 0.02899741567671299, 0.3809700310230255], [0.41907304525375366, 0.3542965352535248, 0.4363647997379303, 0.008938251063227654], [0.04087149351835251, 0.09524080157279968, 0.02899741567671299, 0.3809700310230255], [0.03451932221651077, 0.2708795964717865, 0.3699817359447479, 0.16447588801383972], [0.03451932221651077, 0.2708795964717865, 0.3699817359447479, 0.16447588801383972]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.018672475591301918, 0.12544764578342438, 0.3759167492389679, 0.07687605172395706], [0.4025471806526184, 0.00022576440824195743, 0.48760801553726196, 0.14939065277576447], [0.1845788210630417, 0.4405200481414795, 0.27396348118782043, 0.2582034468650818], [0.4025471806526184, 0.00022576440824195743, 0.48760801553726196, 0.14939065277576447], [0.1845788210630417, 0.4405200481414795, 0.27396348118782043, 0.2582034468650818], [0.31563013792037964, 0.4935934841632843, 0.46372902393341064, 0.46050912141799927], [0.31563013792037964, 0.4935934841632843, 0.46372902393341064, 0.46050912141799927]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_f39018f687dacb5f1098d46033628a87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32261de2d6eb92f1b61d195845c9fba2
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f39018f687dacb5f1098d46033628a87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32261de2d6eb92f1b61d195845c9fba2
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_ab8e86380a31a9bbb091c66d534fea33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_674c962dcbbeff6af94ff9ebb91ba7aa
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_11eb759711b98d20070374f21d13e321(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3daa8676592f86ab4a3245e8d8e439c6
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_6251e449bbf8a7ab400608d721de5490(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44525033235549927, 0.05215751752257347, 0.3241686224937439, 0.16674992442131042, 0.17867626249790192, 0.076868936419487], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2925397753715515, 0.4553702473640442, 0.23048850893974304, 0.27912789583206177, 0.4798333942890167, 0.1506512463092804], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_e3bbfb29b7befee5ee5fb2987adffe63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.35288164019584656, 0.2496812641620636, 0.19527703523635864, 0.02379102259874344, 0.29160597920417786, 0.1866409331560135], dtype='float32').reshape([6]),
            paddle.to_tensor([0.40655165910720825, 0.18528297543525696, 0.49778977036476135, 0.06130501255393028, 0.47947970032691956, 0.3735698163509369], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_aa704fcbaa830b8d381dd36a327a50cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18762415647506714, 0.2747806906700134, 0.4200975298881531, 0.3786199986934662, 0.03297353908419609, 0.23518399894237518], dtype='float32').reshape([6]),
            paddle.to_tensor([0.28722062706947327, 0.21731960773468018, 0.3951013386249542, 0.3923845887184143, 0.3277442157268524, 0.35763120651245117], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_d2d85bd510811fb60043a2e8b61a4ae5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18261484801769257, 0.3417728841304779, 0.2622556686401367, 0.1527484655380249, 0.36947619915008545, 0.32382044196128845], dtype='float32').reshape([6]),
            paddle.to_tensor([0.43829435110092163, 0.012103257700800896, 0.42449864745140076, 0.4616372585296631, 0.0976329892873764, 0.0438818633556366], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_1c0f23a2379149a689c68bdfe3fff68e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18762415647506714, 0.2747806906700134, 0.3241686224937439, 0.27912789583206177, 0.03297353908419609, 0.1506512463092804], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2925397753715515, 0.4553702473640442, 0.3951013386249542, 0.3923845887184143, 0.4798333942890167, 0.35763120651245117], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_174986a862da54cbfe956c86c8dab3a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18261484801769257, 0.2496812641620636, 0.2622556686401367, 0.06130501255393028, 0.36947619915008545, 0.32382044196128845], dtype='float32').reshape([6]),
            paddle.to_tensor([0.43829435110092163, 0.18528297543525696, 0.49778977036476135, 0.4616372585296631, 0.47947970032691956, 0.3735698163509369], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_548e105b4112e99c095b0c79c53374c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44525033235549927, 0.4553702473640442, 0.3241686224937439, 0.27912789583206177, 0.4798333942890167, 0.1506512463092804], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2925397753715515, 0.4553702473640442, 0.23048850893974304, 0.27912789583206177, 0.4798333942890167, 0.1506512463092804], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_a6158a06a3db2de9308be9aa86524f5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.40655165910720825, 0.2496812641620636, 0.49778977036476135, 0.06130501255393028, 0.47947970032691956, 0.3735698163509369], dtype='float32').reshape([6]),
            paddle.to_tensor([0.40655165910720825, 0.18528297543525696, 0.49778977036476135, 0.06130501255393028, 0.47947970032691956, 0.3735698163509369], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_65934aaaf02355776eaeda71ac22154b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.025464775040745735, 0.018943173810839653, -0.0040554567240178585, 0.004251727368682623, -0.0801314041018486, -0.034277696162462234], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, -0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_40747b0282fa34016fd0a87a24207b8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3688950538635254, 0.253763884305954, 0.2773285508155823, 0.2229389101266861, 0.3292548358440399, 0.1137600913643837], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2374223917722702, 0.2460501492023468, 0.40759944915771484, 0.38550227880477905, 0.1803588718175888, 0.2964076101779938], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_cee67cbb1a75f26efd83e8327600bf34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3797166347503662, 0.21748211979866028, 0.3465334177017212, 0.04254801571369171, 0.3855428397655487, 0.2801053822040558], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3104546070098877, 0.17693807184696198, 0.34337717294692993, 0.307192862033844, 0.23355460166931152, 0.18385115265846252], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_596a9489e6c056e8aecf74c54335a8f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44525033235549927, 0.4553702473640442, 0.4200975298881531, 0.3786199986934662, 0.4798333942890167, 0.23518399894237518], dtype='float32').reshape([6]),
            paddle.to_tensor([0.28722062706947327, 0.21731960773468018, 0.23048850893974304, 0.27912789583206177, 0.3277442157268524, 0.1506512463092804], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_85a7331c96b21e91ac2e9987247828f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.40655165910720825, 0.3417728841304779, 0.49778977036476135, 0.1527484655380249, 0.47947970032691956, 0.3735698163509369], dtype='float32').reshape([6]),
            paddle.to_tensor([0.40655165910720825, 0.012103257700800896, 0.42449864745140076, 0.06130501255393028, 0.0976329892873764, 0.0438818633556366], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_90a2e296fdc4c9130818aea961bf7d94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3714536130428314, 0.17256540060043335, -0.152864471077919, 0.0445321761071682, -0.8258402347564697, -0.4123327136039734], dtype='float32').reshape([6]),
            paddle.to_tensor([-1.2328310012817383, -1.4124209880828857, -0.30030757188796997, 1.248608112335205, 1.0130319595336914, 0.37593594193458557], dtype='float32').reshape([6]),
        ]


class PrimitiveOp_c0c6f20bdb7166c96284675ad1b0a67a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1763, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1763, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d390463e25c9ca2a0537d2ee5273007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0c6f20bdb7166c96284675ad1b0a67a
    def get_inputs(self):
        return [
            paddle.uniform([1763, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f328a2f2913336739dce07a7f3f9d998(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1763, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1763, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_648e0786e0c8ddcbb5236e531efdee71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f328a2f2913336739dce07a7f3f9d998
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_648e0786e0c8ddcbb5236e531efdee71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f328a2f2913336739dce07a7f3f9d998
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_648e0786e0c8ddcbb5236e531efdee71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f328a2f2913336739dce07a7f3f9d998
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_648e0786e0c8ddcbb5236e531efdee71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f328a2f2913336739dce07a7f3f9d998
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_648e0786e0c8ddcbb5236e531efdee71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f328a2f2913336739dce07a7f3f9d998
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_648e0786e0c8ddcbb5236e531efdee71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f328a2f2913336739dce07a7f3f9d998
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_648e0786e0c8ddcbb5236e531efdee71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f328a2f2913336739dce07a7f3f9d998
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_648e0786e0c8ddcbb5236e531efdee71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f328a2f2913336739dce07a7f3f9d998
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_648e0786e0c8ddcbb5236e531efdee71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f328a2f2913336739dce07a7f3f9d998
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_648e0786e0c8ddcbb5236e531efdee71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f328a2f2913336739dce07a7f3f9d998
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_648e0786e0c8ddcbb5236e531efdee71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f328a2f2913336739dce07a7f3f9d998
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e71b0a3d00d6453d20564b8fed96ebd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a4c83f203e9f67b723bc396bd4151be
    def get_inputs(self):
        return [
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da5d7e2a679c607dfd94a865246289dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7e07e42ba4884aed6c7a3a1e4ff1f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d390463e25c9ca2a0537d2ee5273007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0c6f20bdb7166c96284675ad1b0a67a
    def get_inputs(self):
        return [
            paddle.uniform([1763, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_ab2a4365c457fb35d67276008dc16ec9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a430d95d326d4c5d5829f45991802fc7
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_9e1e385a7e93ed6204b5a0724734aba4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0df867a4644c834ec9fd270790bdf2c
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
            paddle.to_tensor([0.03305868059396744, 0.10216449201107025, 0.18479007482528687, 0.45377635955810547, 0.12286072969436646, 0.320881724357605, 0.20274046063423157, 0.3258246183395386, 0.29638341069221497, 0.31288883090019226, 0.25943881273269653, 0.19659900665283203, 0.07940669357776642, 0.3430773913860321, 0.4289169907569885, 0.44558948278427124, 0.20496925711631775, 0.14743173122406006, 0.1555706411600113, 0.406779408454895, 0.39422574639320374, 0.0022566039115190506, 0.2913941740989685, 0.3918576240539551], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_29a62f9c2c02e63f0ac0a78afbc53364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0df867a4644c834ec9fd270790bdf2c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03305868059396744, 0.10216449201107025, 0.18479007482528687, 0.45377635955810547, 0.12286072969436646, 0.320881724357605, 0.20274046063423157, 0.3258246183395386, 0.29638341069221497, 0.31288883090019226, 0.25943881273269653, 0.19659900665283203, 0.07940669357776642, 0.3430773913860321, 0.4289169907569885, 0.44558948278427124, 0.20496925711631775, 0.14743173122406006, 0.1555706411600113, 0.406779408454895, 0.39422574639320374, 0.0022566039115190506, 0.2913941740989685, 0.3918576240539551], dtype='float32').reshape([24]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_078bd9ace561cea6e69653214a30bfa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_078bd9ace561cea6e69653214a30bfa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_078bd9ace561cea6e69653214a30bfa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_078bd9ace561cea6e69653214a30bfa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_078bd9ace561cea6e69653214a30bfa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_078bd9ace561cea6e69653214a30bfa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_078bd9ace561cea6e69653214a30bfa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec31835e8609894bd0ca79587cdfc721
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0809e6f04bd84d889e2d6b72d4651ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0809e6f04bd84d889e2d6b72d4651ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0809e6f04bd84d889e2d6b72d4651ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0809e6f04bd84d889e2d6b72d4651ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0809e6f04bd84d889e2d6b72d4651ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0809e6f04bd84d889e2d6b72d4651ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0809e6f04bd84d889e2d6b72d4651ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_202983865b0e3a39199ea44ed1b4584f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9c80c41ab8f1cd76a480326466c451bd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1490, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1490, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1291c0f2583ed5fc23f0b14258aa2f08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c80c41ab8f1cd76a480326466c451bd
    def get_inputs(self):
        return [
            paddle.uniform([1490, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2b83f329a42580a3e33a6853b5718258(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1490, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1490, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_86d924ac78bbd975df44b0dc44df32c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b83f329a42580a3e33a6853b5718258
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86d924ac78bbd975df44b0dc44df32c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b83f329a42580a3e33a6853b5718258
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86d924ac78bbd975df44b0dc44df32c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b83f329a42580a3e33a6853b5718258
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86d924ac78bbd975df44b0dc44df32c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b83f329a42580a3e33a6853b5718258
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86d924ac78bbd975df44b0dc44df32c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b83f329a42580a3e33a6853b5718258
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86d924ac78bbd975df44b0dc44df32c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b83f329a42580a3e33a6853b5718258
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86d924ac78bbd975df44b0dc44df32c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b83f329a42580a3e33a6853b5718258
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86d924ac78bbd975df44b0dc44df32c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b83f329a42580a3e33a6853b5718258
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86d924ac78bbd975df44b0dc44df32c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b83f329a42580a3e33a6853b5718258
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86d924ac78bbd975df44b0dc44df32c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b83f329a42580a3e33a6853b5718258
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86d924ac78bbd975df44b0dc44df32c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b83f329a42580a3e33a6853b5718258
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_8b5d4c4fb067c113314ed54016498a8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_28893411881ab0c201d00b2effafe770
    def get_inputs(self):
        return [
            paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_ae5c0a69a2e1fcf0cb65bc7f659e0058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccd1f9a8924543a5f9c8c87c15c9e66e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1291c0f2583ed5fc23f0b14258aa2f08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c80c41ab8f1cd76a480326466c451bd
    def get_inputs(self):
        return [
            paddle.uniform([1490, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f032e8640c782bd0a305029ba1823bf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f032e8640c782bd0a305029ba1823bf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f032e8640c782bd0a305029ba1823bf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f032e8640c782bd0a305029ba1823bf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f032e8640c782bd0a305029ba1823bf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f032e8640c782bd0a305029ba1823bf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f032e8640c782bd0a305029ba1823bf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5562693c80f4b6c017b21160400867ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f603fd880d44835c63e3968a3927fab1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_473768ad5406d82abfbdd94c8502a0d7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
            paddle.to_tensor([0.11946789920330048, 0.4131564795970917, 0.2905661165714264, 0.21301473677158356], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_7ecc862d6fd55ba925c17a8e6bdcc050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_473768ad5406d82abfbdd94c8502a0d7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11946789920330048, 0.4131564795970917, 0.2905661165714264, 0.21301473677158356], dtype='float32').reshape([4]),
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


class TestPrimitiveOp_db8193251c597154b5bbf986edd68f79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db8193251c597154b5bbf986edd68f79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db8193251c597154b5bbf986edd68f79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db8193251c597154b5bbf986edd68f79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db8193251c597154b5bbf986edd68f79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db8193251c597154b5bbf986edd68f79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db8193251c597154b5bbf986edd68f79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dabcdad19e17062c4224151836e2997
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_93bf25610fd14d81a6b319a42245208a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_71b5534acaa14cda127c5d6eb7dce1a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93bf25610fd14d81a6b319a42245208a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.49306923151016235, 0.19773545861244202, 0.09421505033969879, 0.1957722008228302], [0.09745010733604431, 0.18165327608585358, 0.4203101694583893, 0.06101805344223976], [0.46258634328842163, 0.3231339454650879, 0.3041573762893677, 0.40647998452186584], [0.40171483159065247, 0.10056062042713165, 0.4877328872680664, 0.1391730010509491], [0.40171483159065247, 0.10056062042713165, 0.4877328872680664, 0.1391730010509491], [0.46258634328842163, 0.3231339454650879, 0.3041573762893677, 0.40647998452186584]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.02088022604584694, 0.05845842510461807, 0.4237040579319, 0.08926024287939072], [0.09806802868843079, 0.17826540768146515, 0.3628913462162018, 0.11528350412845612], [0.08538620173931122, 0.24740161001682281, 0.24447958171367645, 0.39610692858695984], [0.174465611577034, 0.2589118480682373, 0.20285987854003906, 0.3282819390296936], [0.174465611577034, 0.2589118480682373, 0.20285987854003906, 0.3282819390296936], [0.08538620173931122, 0.24740161001682281, 0.24447958171367645, 0.39610692858695984]], dtype='float32').reshape([6, 4]),
        ]


class TestPrimitiveOp_b51d6d05855d342654a6841cefa85a6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_012768c0db613e5ecd733a6a0e0edd3f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.356364369392395, 0.3296383023262024, 0.1307453066110611, 0.0343102402985096], [0.16923296451568604, 0.05281924083828926, 0.18031005561351776, 0.02500438503921032], [0.11964826285839081, 0.07826577872037888, 0.15798108279705048, 0.02544194646179676], [0.2868961989879608, 0.10892890393733978, 0.4145793914794922, 0.3044985830783844], [0.356364369392395, 0.3296383023262024, 0.1307453066110611, 0.0343102402985096]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.4480953812599182, 0.17784923315048218, 0.17060598731040955, 0.037088543176651], [0.4193055033683777, 0.1312147080898285, 0.01361282728612423, 0.3651732802391052], [0.15718525648117065, 0.299811452627182, 0.4366101622581482, 0.3562086820602417], [0.1485888659954071, 0.4197021722793579, 0.08429938554763794, 0.44268089532852173], [0.4480953812599182, 0.17784923315048218, 0.17060598731040955, 0.037088543176651]], dtype='float32').reshape([5, 4]),
        ]


class PrimitiveOp_a6ab415439ce28de9bd2e68cea09d44a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d45e2a0209016a31ef90a3ab417cc8bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6ab415439ce28de9bd2e68cea09d44a
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_b5d1b84d45b654fb7488e077e35e901b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3290674686431885]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.491269052028656]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_589950d70e26230eea42d72b0f1a683d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0780901163816452]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.09474091231822968]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_9b02a0a459af1c8086ac5ecd8aec00c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.36167198419570923]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.491269052028656]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_5c080d9ddc82192125467e85fcbe6fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1334301382303238]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.09417319297790527]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_0ade3ec6331bf944abc3d93255ad6f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3290674686431885]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.4348372220993042]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_589950d70e26230eea42d72b0f1a683d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0780901163816452]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.09474091231822968]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_d9034decc9da50d05137f6e3cf642292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0033264346420764923]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_1c820f3a7717718d8443952128d008d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.36167198419570923]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.4348372220993042]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_5c080d9ddc82192125467e85fcbe6fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1334301382303238]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.09417319297790527]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_4a6f72531906d74ac45d222297e79f9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.002872243756428361]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.0033264346420764923]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_0bb8c8d5f54aa257cbc1ca0a10f9c6d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.15813103318214417]], dtype='float32').reshape([1, 1]),
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


class TestPrimitiveOp_332f9ce1b2331c693bace80710d4b948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3079056143760681], [0.17066887021064758], [0.32239457964897156], [0.12811346352100372], [0.056965190917253494], [0.28739556670188904]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.49020639061927795], [0.4823766052722931], [0.3955044746398926], [0.22639241814613342], [0.3185456097126007], [0.4976474642753601]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_40f9f7dbe87f590a128448dc56f55089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14837408065795898], [0.1733187437057495], [0.2672334313392639], [0.09106913208961487], [0.018445204943418503], [0.12337625026702881]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.38628971576690674], [0.3818877637386322], [0.4154316186904907], [0.09654207527637482], [0.48540663719177246], [0.23541131615638733]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_ab98e777273795df3250981a6b4ab35d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.40875673294067383], [0.17066887021064758], [0.32239457964897156], [0.12811346352100372], [0.056965190917253494], [0.37778374552726746]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.49020639061927795], [0.06694857776165009], [0.37551629543304443], [0.16441957652568817], [0.056271616369485855], [0.4976474642753601]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_c71a01df48ce5c75486c5ee68e60aba9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14837408065795898], [0.1733187437057495], [0.2672334313392639], [0.09106913208961487], [0.15825320780277252], [0.12337625026702881]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.38628971576690674], [0.3818877637386322], [0.4154316186904907], [0.09654207527637482], [0.48540663719177246], [0.23541131615638733]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_e987219f6568be82953b3b70c5af5916(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3079056143760681], [0.3012850880622864], [0.45780348777770996], [0.38056784868240356], [0.45184314250946045], [0.28739556670188904]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.03940660506486893], [0.4823766052722931], [0.3955044746398926], [0.22639241814613342], [0.3185456097126007], [0.04834653064608574]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_8c3ac6bda23af960e6b07d6bc7716f33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.23708929121494293], [0.3412092924118042], [0.4712662696838379], [0.33640754222869873], [0.018445204943418503], [0.33481597900390625]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.13275480270385742], [0.24877241253852844], [0.14286577701568604], [0.02221701852977276], [0.3493429720401764], [0.074555404484272]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_db2c613e3c620df2a930d04deff83fb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0473918542265892], [-0.03837237507104874], [0.028331568464636803], [0.04863915964961052], [-0.044334761798381805], [0.07564397901296616]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_68632f0f7987a98960fef5fe3b9a15a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.40875673294067383], [0.3012850880622864], [0.45780348777770996], [0.38056784868240356], [0.45184314250946045], [0.37778374552726746]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.03940660506486893], [0.06694857776165009], [0.37551629543304443], [0.16441957652568817], [0.056271616369485855], [0.04834653064608574]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_c01064af9a7cae74f47b030da31aa998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.23708929121494293], [0.3412092924118042], [0.4712662696838379], [0.33640754222869873], [0.15825320780277252], [0.33481597900390625]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.13275480270385742], [0.24877241253852844], [0.14286577701568604], [0.02221701852977276], [0.3493429720401764], [0.074555404484272]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_8490ea6ccc65a7fcc31747cada0f75e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0385359562933445], [0.021661335602402687], [0.027023155242204666], [0.06791174411773682], [-0.07558967173099518], [0.08573952317237854]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0473918542265892], [-0.03837237507104874], [0.028331568464636803], [0.04863915964961052], [-0.044334761798381805], [0.07564397901296616]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_f7038c7db638a1d0e4895f8e990eb665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [-0.0], [0.0], [0.0], [-0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.22980870306491852], [2.7714684009552], [-0.04841822385787964], [0.28378868103027344], [0.4134812355041504], [0.11774668097496033]], dtype='float32').reshape([6, 1]),
        ]


class PrimitiveOp_1e223e011a18f61a428db8a806582d14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_081d1ef996afe9a87a0e2f2c6e0c1062(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e223e011a18f61a428db8a806582d14
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.37872758507728577, 0.37213441729545593, 0.3093475103378296, 0.3466503918170929], [0.2208942025899887, 0.21938829123973846, 0.2680375576019287, 0.02463166043162346], [0.27107474207878113, 0.19161130487918854, 0.1335662603378296, 0.08675449341535568], [0.22501160204410553, 0.07493574172258377, 0.4872724711894989, 0.13434165716171265]], dtype='float32').reshape([4, 4]),
            paddle.to_tensor([[0.10332943499088287, 0.09419811517000198, 0.1365530639886856, 0.4799371361732483], [0.31450971961021423, 0.36414796113967896, 0.18038342893123627, 0.3522614538669586], [0.4420488476753235, 0.46016988158226013, 0.13171501457691193, 0.3975575566291809], [0.09566506743431091, 0.07718977332115173, 0.2090953141450882, 0.4393099844455719]], dtype='float32').reshape([4, 4]),
        ]


class TestPrimitiveOp_96c1c418aa2419e5974f8b26da406b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96c1c418aa2419e5974f8b26da406b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96c1c418aa2419e5974f8b26da406b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96c1c418aa2419e5974f8b26da406b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96c1c418aa2419e5974f8b26da406b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96c1c418aa2419e5974f8b26da406b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96c1c418aa2419e5974f8b26da406b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cad50791374cc1aa03b5e99df27d987
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_693a35f419aeac89656bf62152a0072f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_693a35f419aeac89656bf62152a0072f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_693a35f419aeac89656bf62152a0072f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_693a35f419aeac89656bf62152a0072f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_693a35f419aeac89656bf62152a0072f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_693a35f419aeac89656bf62152a0072f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_693a35f419aeac89656bf62152a0072f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_992c1c7449619f449fe668a2146b44c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_11b85bde8e53afe96e4793161614ce3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_206da4c03be6a988c08e2f59f6306763
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6f1a9a18bae4ac29cdffaf5f31148535(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2076, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2076, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53543560959389f8a1b4245837b03d44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1a9a18bae4ac29cdffaf5f31148535
    def get_inputs(self):
        return [
            paddle.uniform([2076, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6914ff7b89549258067e83b2ef243901(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2076, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2076, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f6b93a4f9e0010ede734b09dc1ba493(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6914ff7b89549258067e83b2ef243901
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f6b93a4f9e0010ede734b09dc1ba493(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6914ff7b89549258067e83b2ef243901
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f6b93a4f9e0010ede734b09dc1ba493(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6914ff7b89549258067e83b2ef243901
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f6b93a4f9e0010ede734b09dc1ba493(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6914ff7b89549258067e83b2ef243901
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f6b93a4f9e0010ede734b09dc1ba493(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6914ff7b89549258067e83b2ef243901
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f6b93a4f9e0010ede734b09dc1ba493(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6914ff7b89549258067e83b2ef243901
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f6b93a4f9e0010ede734b09dc1ba493(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6914ff7b89549258067e83b2ef243901
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f6b93a4f9e0010ede734b09dc1ba493(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6914ff7b89549258067e83b2ef243901
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f6b93a4f9e0010ede734b09dc1ba493(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6914ff7b89549258067e83b2ef243901
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f6b93a4f9e0010ede734b09dc1ba493(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6914ff7b89549258067e83b2ef243901
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f6b93a4f9e0010ede734b09dc1ba493(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6914ff7b89549258067e83b2ef243901
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_91e7af10d4ee20f1e8f7e3a5df630b35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50398eaf8f3d9b9c285466c003f7bb78
    def get_inputs(self):
        return [
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_c15f99175a74d6e38f2b36214f210aa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0808c9fcbc960a398d51addf53e19cab
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53543560959389f8a1b4245837b03d44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f1a9a18bae4ac29cdffaf5f31148535
    def get_inputs(self):
        return [
            paddle.uniform([2076, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d089b59d5a99376312423f58bda59928(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2f2b4663ad0e38d2e5e931474ae84ac
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.26236623525619507, 0.1946115791797638, 0.4438350796699524, 0.1940588802099228], [0.26236623525619507, 0.1946115791797638, 0.4438350796699524, 0.1940588802099228], [0.4003095328807831, 0.310364693403244, 0.346569687128067, 0.015020975843071938], [0.13913795351982117, 0.10304193198680878, 0.3325067162513733, 0.3271404206752777], [0.03147411346435547, 0.31711843609809875, 0.35153627395629883, 0.35504013299942017], [0.16475170850753784, 0.43198397755622864, 0.11937560886144638, 0.053075727075338364], [0.07177449762821198, 0.3370377719402313, 0.032082319259643555, 0.2462560534477234]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.22727954387664795, 0.3631150424480438, 0.25423330068588257, 0.18690410256385803], [0.22727954387664795, 0.3631150424480438, 0.25423330068588257, 0.18690410256385803], [0.29397067427635193, 0.2375965267419815, 0.3166613280773163, 0.13694150745868683], [0.07406700402498245, 0.3467102646827698, 0.12178514897823334, 0.24275782704353333], [0.14555078744888306, 0.31605976819992065, 0.04637580364942551, 0.09301868081092834], [0.09601700305938721, 0.42892199754714966, 0.06740482896566391, 0.14872956275939941], [0.33403050899505615, 0.27293017506599426, 0.1705905795097351, 0.20263902842998505]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_b1a7a567b281e3c5506f25971e9b80aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1a7a567b281e3c5506f25971e9b80aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1a7a567b281e3c5506f25971e9b80aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1a7a567b281e3c5506f25971e9b80aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1a7a567b281e3c5506f25971e9b80aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1a7a567b281e3c5506f25971e9b80aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1a7a567b281e3c5506f25971e9b80aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_8e54265e1af59e115fe0392f67de0d31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b630987069d3aa1437c668cf0e1bc8e0
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_311f870d8ecbc4db5cc8aa3d6fbdd0e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d65965f879918926c79768cf3a2702ba
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_48cb9a1d3b148db6a58311b8d5804a60(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4642, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[4642, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0a373c0ed36de03cf6e12822059ea20c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48cb9a1d3b148db6a58311b8d5804a60
    def get_inputs(self):
        return [
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d1fd54f3e854896eebcb0ecd8afc75a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4642, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4642, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1d463793d01a1cf01ce2ff6c161be3b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1fd54f3e854896eebcb0ecd8afc75a4
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d463793d01a1cf01ce2ff6c161be3b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1fd54f3e854896eebcb0ecd8afc75a4
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d463793d01a1cf01ce2ff6c161be3b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1fd54f3e854896eebcb0ecd8afc75a4
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d463793d01a1cf01ce2ff6c161be3b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1fd54f3e854896eebcb0ecd8afc75a4
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d463793d01a1cf01ce2ff6c161be3b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1fd54f3e854896eebcb0ecd8afc75a4
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d463793d01a1cf01ce2ff6c161be3b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1fd54f3e854896eebcb0ecd8afc75a4
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d463793d01a1cf01ce2ff6c161be3b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1fd54f3e854896eebcb0ecd8afc75a4
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d463793d01a1cf01ce2ff6c161be3b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1fd54f3e854896eebcb0ecd8afc75a4
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d463793d01a1cf01ce2ff6c161be3b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1fd54f3e854896eebcb0ecd8afc75a4
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d463793d01a1cf01ce2ff6c161be3b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1fd54f3e854896eebcb0ecd8afc75a4
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d463793d01a1cf01ce2ff6c161be3b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1fd54f3e854896eebcb0ecd8afc75a4
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_1eebb81d21e35473007b2b5b7e1f0087(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f39622dcf124217b362e6da2a03f1af
    def get_inputs(self):
        return [
            paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_e2ef749d596d65350bc534e340d23a68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a034f0558f586eb2751e2cf8dc120268
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a373c0ed36de03cf6e12822059ea20c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48cb9a1d3b148db6a58311b8d5804a60
    def get_inputs(self):
        return [
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ee1d09c76b679c656ebd15cf34707190(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1047, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1047, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aba56d6cc882af23bfc7e4bf920b0408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee1d09c76b679c656ebd15cf34707190
    def get_inputs(self):
        return [
            paddle.uniform([1047, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0d3ae1dcabb7c41959c7c0219e7db78a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1047, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1047, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4558b7c1bc9e0e42b4c87898f30c9190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3ae1dcabb7c41959c7c0219e7db78a
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4558b7c1bc9e0e42b4c87898f30c9190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3ae1dcabb7c41959c7c0219e7db78a
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4558b7c1bc9e0e42b4c87898f30c9190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3ae1dcabb7c41959c7c0219e7db78a
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4558b7c1bc9e0e42b4c87898f30c9190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3ae1dcabb7c41959c7c0219e7db78a
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4558b7c1bc9e0e42b4c87898f30c9190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3ae1dcabb7c41959c7c0219e7db78a
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4558b7c1bc9e0e42b4c87898f30c9190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3ae1dcabb7c41959c7c0219e7db78a
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4558b7c1bc9e0e42b4c87898f30c9190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3ae1dcabb7c41959c7c0219e7db78a
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4558b7c1bc9e0e42b4c87898f30c9190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3ae1dcabb7c41959c7c0219e7db78a
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4558b7c1bc9e0e42b4c87898f30c9190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3ae1dcabb7c41959c7c0219e7db78a
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4558b7c1bc9e0e42b4c87898f30c9190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3ae1dcabb7c41959c7c0219e7db78a
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4558b7c1bc9e0e42b4c87898f30c9190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3ae1dcabb7c41959c7c0219e7db78a
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_ff3dc64564fe40d9cad5441eb1db6982(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0386906bae6ceafa65c95a5bb420c828
    def get_inputs(self):
        return [
            paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_74a2209a424a61e2152533a0aa6058ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_639ea61878bad07af55eb84bbc394179
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aba56d6cc882af23bfc7e4bf920b0408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee1d09c76b679c656ebd15cf34707190
    def get_inputs(self):
        return [
            paddle.uniform([1047, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_2f25327228edf7f27ef0f72b0a06ffc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38aa68f578e2965a292222c0287d8dc6
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ccb15cd2897401003de1af591299701a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93bf25610fd14d81a6b319a42245208a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.31197309494018555, 0.48581066727638245, 0.2665523886680603, 0.3237149715423584], [0.12674857676029205, 0.49678468704223633, 0.010301018133759499, 0.2673981189727783], [0.12674857676029205, 0.49678468704223633, 0.010301018133759499, 0.2673981189727783], [0.1253647655248642, 0.44840967655181885, 0.007888504303991795, 0.18193991482257843], [0.4002373218536377, 0.3019247353076935, 0.025422099977731705, 0.49035099148750305], [0.024372011423110962, 0.3659110367298126, 0.16841275990009308, 0.1677943766117096]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.11913707107305527, 0.039375923573970795, 0.3931671679019928, 0.0066429139114916325], [0.1816796213388443, 0.2070293426513672, 0.22162753343582153, 0.3538522720336914], [0.1816796213388443, 0.2070293426513672, 0.22162753343582153, 0.3538522720336914], [0.46572351455688477, 0.10193134099245071, 0.08952648192644119, 0.18998749554157257], [0.15245385468006134, 0.4078909158706665, 0.3399084806442261, 0.35077860951423645], [0.4055851101875305, 0.10484957695007324, 0.3074356019496918, 0.34920889139175415]], dtype='float32').reshape([6, 4]),
        ]


class TestPrimitiveOp_bc1907066b2d16a9a9ec8d014f531d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc1907066b2d16a9a9ec8d014f531d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc1907066b2d16a9a9ec8d014f531d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc1907066b2d16a9a9ec8d014f531d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc1907066b2d16a9a9ec8d014f531d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc1907066b2d16a9a9ec8d014f531d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc1907066b2d16a9a9ec8d014f531d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dbc63f026b0b0c7911474c2f678e05d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_59e4f896a616a4472bb9d02e25783ade(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6131f1e087763b89c98fd9eaba27d305
    def get_inputs(self):
        return [
            paddle.uniform([100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 2], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_ed137d3118c851cbb429cc3d29674558(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1546d5060eb5e7f91c606a3154f7d7bd
    def get_inputs(self):
        return [
            paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1.0968555212020874, 1.0203711986541748, 0.5792708396911621, 1.1560919284820557], [10.179582595825195, 1.6009176969528198, 0.9537999033927917, 0.48562008142471313]], dtype='float32').reshape([2, 4]),
        ]


class TestPrimitiveOp_1dc12a1fada6abd96fa33923fd8f8858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1dc12a1fada6abd96fa33923fd8f8858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1dc12a1fada6abd96fa33923fd8f8858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1dc12a1fada6abd96fa33923fd8f8858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1dc12a1fada6abd96fa33923fd8f8858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1dc12a1fada6abd96fa33923fd8f8858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1dc12a1fada6abd96fa33923fd8f8858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e357b570657f6fab550c03fc9bba497e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_09e8a17dc1682601f9783fa0972d204f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3fabb513d00823b46cfe60432a80317
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_c48e9a8dbe983041d2b91b60f8076681(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef80528f0a858e03e82f793ae6527930
    def get_inputs(self):
        return [
            paddle.uniform([300, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 2], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_afd473d8f079723387426cd7c224a570(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29eb514882c92ab3a86517239fc1edf9
    def get_inputs(self):
        return [
            paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.36238473653793335, 1.562929630279541, 3.921856641769409, 0.2439769059419632], [0.1412999927997589, 1.4040974378585815, 1.5319933891296387, 1.0357928276062012]], dtype='float32').reshape([2, 4]),
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


class TestPrimitiveOp_ae1803171006f924321f925a443494f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12497831881046295], [0.2808596193790436], [0.11296632140874863], [0.20338411629199982], [0.2920495271682739]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.28228238224983215], [0.4601689577102661], [0.23982450366020203], [0.4331777095794678], [0.20875117182731628]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_1280faba20094553192f7984f884a19f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4368440806865692], [0.25516924262046814], [0.033184755593538284], [0.07404229044914246], [0.4006950557231903]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.4062413275241852], [0.36561113595962524], [0.4665828347206116], [0.33214429020881653], [0.37686073780059814]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_ab867c8698053b5d85e4a9798b39eca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12497831881046295], [0.2808596193790436], [0.11296632140874863], [0.20338411629199982], [0.2920495271682739]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.15261472761631012], [0.34060344099998474], [0.2098308652639389], [0.34933146834373474], [0.20875117182731628]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_b4e862aed4827871fa3483214e0c79a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4368440806865692], [0.25516924262046814], [0.033184755593538284], [0.33212995529174805], [0.40790942311286926]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.05367068201303482], [0.36561113595962524], [0.4665828347206116], [0.2871648371219635], [0.07508329302072525]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_39a24450e96750ce120b11e4620a2601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39957284927368164], [0.406246155500412], [0.24995972216129303], [0.3918718695640564], [0.4563639461994171]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.28228238224983215], [0.4601689577102661], [0.23982450366020203], [0.4331777095794678], [0.05244790017604828]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_43a2d83fc094ea11469ca9318249bdb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.47180286049842834], [0.31962135434150696], [0.32587990164756775], [0.07404229044914246], [0.4006950557231903]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.4062413275241852], [0.011468961834907532], [0.3539048135280609], [0.33214429020881653], [0.37686073780059814]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_caa1745eabc4a96d06e4c800bf775a50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0028997943736612797], [-0.01001821830868721], [0.04169686883687973], [0.004098579753190279], [0.037350933998823166]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0019853594712913036]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_42bea5decbae2e1526d72736100e918a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39957284927368164], [0.406246155500412], [0.24995972216129303], [0.3918718695640564], [0.4563639461994171]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.15261472761631012], [0.34060344099998474], [0.2098308652639389], [0.34933146834373474], [0.05244790017604828]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_01b4095c8ac9cf026b37db51846d2210(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.47180286049842834], [0.31962135434150696], [0.32587990164756775], [0.33212995529174805], [0.40790942311286926]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.05367068201303482], [0.011468961834907532], [0.3539048135280609], [0.2871648371219635], [0.07508329302072525]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_61e7df0e0c8c56786067c2b18f0106dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10326113551855087], [0.020227959379553795], [-0.0011246075155213475], [0.001912834239192307], [0.1344338208436966]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.0028997943736612797], [-0.01001821830868721], [0.04169686883687973], [0.004098579753190279], [0.035365574061870575]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_a45bea6fa325b0e544c5b09e3abdb09d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [-0.0], [0.0], [0.0], [0.05613819509744644]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[1.0280821323394775], [1.4952658414840698], [38.076820373535156], [-1.1426738500595093], [0.7369294762611389]], dtype='float32').reshape([5, 1]),
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


class TestPrimitiveOp_0b78777c920f6383865171f9d0e4b329(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bed1e775ba8f7d7f86dd15571017dbbd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db53f78b6790f8202abc17d46b7545d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db53f78b6790f8202abc17d46b7545d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db53f78b6790f8202abc17d46b7545d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db53f78b6790f8202abc17d46b7545d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db53f78b6790f8202abc17d46b7545d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db53f78b6790f8202abc17d46b7545d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db53f78b6790f8202abc17d46b7545d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feea3775d2244a236caefac5af9726fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a5e07c698195cbd639733e10dbe90ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a5e07c698195cbd639733e10dbe90ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a5e07c698195cbd639733e10dbe90ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a5e07c698195cbd639733e10dbe90ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a5e07c698195cbd639733e10dbe90ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a5e07c698195cbd639733e10dbe90ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a5e07c698195cbd639733e10dbe90ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e18eb2ea73687ca99e17be52d48e8b65
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_79eac68b645986b64b21cf7bb9806364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79eac68b645986b64b21cf7bb9806364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79eac68b645986b64b21cf7bb9806364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79eac68b645986b64b21cf7bb9806364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79eac68b645986b64b21cf7bb9806364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79eac68b645986b64b21cf7bb9806364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79eac68b645986b64b21cf7bb9806364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c429af5fe58df95e7a0edd1ffb040503(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2359, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2359, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e161730e6f6c025e57efefd3c35fbea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c429af5fe58df95e7a0edd1ffb040503
    def get_inputs(self):
        return [
            paddle.uniform([2359, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_519cfde3cb37f37630464e3f742f8262(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2359, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2359, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c580dc70a41afcf0b04f0ac1b6cfde5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_519cfde3cb37f37630464e3f742f8262
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c580dc70a41afcf0b04f0ac1b6cfde5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_519cfde3cb37f37630464e3f742f8262
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c580dc70a41afcf0b04f0ac1b6cfde5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_519cfde3cb37f37630464e3f742f8262
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c580dc70a41afcf0b04f0ac1b6cfde5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_519cfde3cb37f37630464e3f742f8262
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c580dc70a41afcf0b04f0ac1b6cfde5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_519cfde3cb37f37630464e3f742f8262
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c580dc70a41afcf0b04f0ac1b6cfde5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_519cfde3cb37f37630464e3f742f8262
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c580dc70a41afcf0b04f0ac1b6cfde5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_519cfde3cb37f37630464e3f742f8262
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c580dc70a41afcf0b04f0ac1b6cfde5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_519cfde3cb37f37630464e3f742f8262
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c580dc70a41afcf0b04f0ac1b6cfde5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_519cfde3cb37f37630464e3f742f8262
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c580dc70a41afcf0b04f0ac1b6cfde5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_519cfde3cb37f37630464e3f742f8262
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c580dc70a41afcf0b04f0ac1b6cfde5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_519cfde3cb37f37630464e3f742f8262
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_ad16ee5eb22f6fcf42de269d9abc7f30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0bccc3bedd14bdf0e2dad1e1f7dbee5
    def get_inputs(self):
        return [
            paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_e10c1016c586313ef4326dc4e5b12935(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdbccc0bd9d4fa6fab63f7856283dda3
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e161730e6f6c025e57efefd3c35fbea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c429af5fe58df95e7a0edd1ffb040503
    def get_inputs(self):
        return [
            paddle.uniform([2359, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4ec0cca8ec5b5e3168ec1088cabb25f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3049, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[3049, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b96158a906a3e4b614516dbffc1ba57e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ec0cca8ec5b5e3168ec1088cabb25f5
    def get_inputs(self):
        return [
            paddle.uniform([3049, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ea773f6fbc009b5a47ca1eb79078f1ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3049, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3049, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92ca4c541cb6f8823463f9d800d2d5af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea773f6fbc009b5a47ca1eb79078f1ab
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92ca4c541cb6f8823463f9d800d2d5af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea773f6fbc009b5a47ca1eb79078f1ab
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92ca4c541cb6f8823463f9d800d2d5af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea773f6fbc009b5a47ca1eb79078f1ab
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92ca4c541cb6f8823463f9d800d2d5af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea773f6fbc009b5a47ca1eb79078f1ab
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92ca4c541cb6f8823463f9d800d2d5af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea773f6fbc009b5a47ca1eb79078f1ab
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92ca4c541cb6f8823463f9d800d2d5af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea773f6fbc009b5a47ca1eb79078f1ab
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92ca4c541cb6f8823463f9d800d2d5af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea773f6fbc009b5a47ca1eb79078f1ab
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92ca4c541cb6f8823463f9d800d2d5af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea773f6fbc009b5a47ca1eb79078f1ab
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92ca4c541cb6f8823463f9d800d2d5af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea773f6fbc009b5a47ca1eb79078f1ab
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92ca4c541cb6f8823463f9d800d2d5af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea773f6fbc009b5a47ca1eb79078f1ab
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92ca4c541cb6f8823463f9d800d2d5af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea773f6fbc009b5a47ca1eb79078f1ab
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f202ef5bf2b7ebbf71a573f1b973a43b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2337ffc18b23e5e45ccf0feb0356b33
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_fc1b85b01c7435fcd4ca58af9a384482(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6eeea1fc78dff7dc1bbecfeb8f6a86ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b96158a906a3e4b614516dbffc1ba57e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ec0cca8ec5b5e3168ec1088cabb25f5
    def get_inputs(self):
        return [
            paddle.uniform([3049, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_20ce128373dc4028baad44a7cfaadcf2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3806, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[3806, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_def7e55d5226f816b481a05957014348(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20ce128373dc4028baad44a7cfaadcf2
    def get_inputs(self):
        return [
            paddle.uniform([3806, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0e8ad4d843085c7ae61284f14dcfed8b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3806, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3806, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_34fe3775b97742304759e61acd4d6ad3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e8ad4d843085c7ae61284f14dcfed8b
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34fe3775b97742304759e61acd4d6ad3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e8ad4d843085c7ae61284f14dcfed8b
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34fe3775b97742304759e61acd4d6ad3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e8ad4d843085c7ae61284f14dcfed8b
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34fe3775b97742304759e61acd4d6ad3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e8ad4d843085c7ae61284f14dcfed8b
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34fe3775b97742304759e61acd4d6ad3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e8ad4d843085c7ae61284f14dcfed8b
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34fe3775b97742304759e61acd4d6ad3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e8ad4d843085c7ae61284f14dcfed8b
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34fe3775b97742304759e61acd4d6ad3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e8ad4d843085c7ae61284f14dcfed8b
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34fe3775b97742304759e61acd4d6ad3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e8ad4d843085c7ae61284f14dcfed8b
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34fe3775b97742304759e61acd4d6ad3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e8ad4d843085c7ae61284f14dcfed8b
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34fe3775b97742304759e61acd4d6ad3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e8ad4d843085c7ae61284f14dcfed8b
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34fe3775b97742304759e61acd4d6ad3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e8ad4d843085c7ae61284f14dcfed8b
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_db7d956918627f3a63aa171cb2042b3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4319ca246764953320ce6e3f935bf6b5
    def get_inputs(self):
        return [
            paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_13bac8d8736ee52ab0e9fcfe234cfd23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33493f3e3759817ddee21e0107f59caf
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_def7e55d5226f816b481a05957014348(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20ce128373dc4028baad44a7cfaadcf2
    def get_inputs(self):
        return [
            paddle.uniform([3806, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_20b118b1ce831beab34235a53ee8fa04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb0e222865ae94b855382033ed43e1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_1f5aed0cd24052e7ff8a0c88e01391bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73583fca9150233dc45d91eabd225d9c
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1a7a567b281e3c5506f25971e9b80aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1a7a567b281e3c5506f25971e9b80aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1a7a567b281e3c5506f25971e9b80aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1a7a567b281e3c5506f25971e9b80aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1a7a567b281e3c5506f25971e9b80aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1a7a567b281e3c5506f25971e9b80aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1a7a567b281e3c5506f25971e9b80aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dceb4d83ca6c7ea940c6f0a4a1d68ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_56223c566930865adafeaa1163b062bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30894a70778990ba804af814e417d17d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_ddf58ae75bc4999f66d1759a1318878a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_042c31aed200f7f6f007e22c52c71138
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
            paddle.to_tensor([0.3323947489261627, 0.055248551070690155, 0.43288254737854004, 0.13294614851474762, 0.20939065515995026, 0.06673471629619598, 0.2870861291885376, 0.03589070588350296, 0.1562991589307785, 0.4208308756351471, 0.20852304995059967, 0.460390567779541, 0.4633416533470154, 0.2641811668872833, 0.35603874921798706, 0.46736380457878113, 0.3329882323741913, 0.4767763018608093, 0.2628551721572876, 0.24535053968429565], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_8d1d42ed457832b6be618d8394a81f7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_042c31aed200f7f6f007e22c52c71138
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3323947489261627, 0.055248551070690155, 0.43288254737854004, 0.13294614851474762, 0.20939065515995026, 0.06673471629619598, 0.2870861291885376, 0.03589070588350296, 0.1562991589307785, 0.4208308756351471, 0.20852304995059967, 0.460390567779541, 0.4633416533470154, 0.2641811668872833, 0.35603874921798706, 0.46736380457878113, 0.3329882323741913, 0.4767763018608093, 0.2628551721572876, 0.24535053968429565], dtype='float32').reshape([20]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([20]),
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


class TestPrimitiveOp_6ec35aa0e8f89ae0521c13c3a9c181b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16607213020324707], [0.1927356868982315], [0.15653924643993378], [0.1810394525527954]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.15882641077041626], [0.06995806097984314], [0.4809706211090088], [0.22571998834609985]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_b33dd5397af05fd3da73fcd32651dee7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05042451620101929], [0.23747287690639496], [0.02164270356297493], [0.027551813051104546]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.43369781970977783], [0.43579286336898804], [0.34903889894485474], [0.4193747043609619]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_c844b2cff7f1157ee69508fb4c53b806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20518168807029724], [0.4319309592247009], [0.47744256258010864], [0.42946767807006836]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0391080342233181], [0.06995806097984314], [0.4809706211090088], [0.22571998834609985]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_deacbf021910c868268fef97ca0fe855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.48733291029930115], [0.3396559953689575], [0.02164270356297493], [0.22589930891990662]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.07258392870426178], [0.43579286336898804], [0.34903889894485474], [0.37684279680252075]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_aed45a9e31939b9a6bc97aaaa459df3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16607213020324707], [0.1927356868982315], [0.15653924643993378], [0.1810394525527954]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.15882641077041626], [0.048668403178453445], [0.17359651625156403], [0.05119822919368744]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_056dfdbc7f67d415fc86c0d4863ba1d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05042451620101929], [0.23747287690639496], [0.43582987785339355], [0.027551813051104546]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.43369781970977783], [0.18939900398254395], [0.3099403977394104], [0.4193747043609619]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_3a7701b746b1ec473013e7ba26f059ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06610178202390671], [-0.027873069047927856], [-0.0009922579629346728], [-0.08162915706634521]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_30452014b5888ef14cb30639339ac371(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20518168807029724], [0.4319309592247009], [0.47744256258010864], [0.42946767807006836]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0391080342233181], [0.048668403178453445], [0.17359651625156403], [0.05119822919368744]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_dfbef90643e427734047b1287c279b4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.48733291029930115], [0.3396559953689575], [0.43582987785339355], [0.22589930891990662]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.07258392870426178], [0.18939900398254395], [0.3099403977394104], [0.37684279680252075]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_2ab7e7f08fb2b0f58def8a4cadf87778(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06887887418270111], [0.0575878769159317], [0.03825102373957634], [-0.057097308337688446]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.06610178202390671], [-0.027873069047927856], [-0.000992257846519351], [-0.08162915706634521]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_48b7ce09ba74524f96d51fc09373bfdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [-0.0], [-0.0], [-0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.04031848907470703], [1.4840092658996582], [1.0259406566619873], [-0.42964982986450195]], dtype='float32').reshape([4, 1]),
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


class TestPrimitiveOp_0535772cc7295d85770b3f300bc01b56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddcc881c03fd6c994972310f815a76b3
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fa800e5f43222004df500bee2775470c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2054, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2054, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2561556861555eda4cf90cdba3362c65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa800e5f43222004df500bee2775470c
    def get_inputs(self):
        return [
            paddle.uniform([2054, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_61bd1e373f9845aca915c1b1d1456713(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2054, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2054, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d77143665bb1ea17cefea8d42d80003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61bd1e373f9845aca915c1b1d1456713
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d77143665bb1ea17cefea8d42d80003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61bd1e373f9845aca915c1b1d1456713
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d77143665bb1ea17cefea8d42d80003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61bd1e373f9845aca915c1b1d1456713
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d77143665bb1ea17cefea8d42d80003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61bd1e373f9845aca915c1b1d1456713
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d77143665bb1ea17cefea8d42d80003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61bd1e373f9845aca915c1b1d1456713
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d77143665bb1ea17cefea8d42d80003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61bd1e373f9845aca915c1b1d1456713
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d77143665bb1ea17cefea8d42d80003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61bd1e373f9845aca915c1b1d1456713
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d77143665bb1ea17cefea8d42d80003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61bd1e373f9845aca915c1b1d1456713
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d77143665bb1ea17cefea8d42d80003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61bd1e373f9845aca915c1b1d1456713
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d77143665bb1ea17cefea8d42d80003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61bd1e373f9845aca915c1b1d1456713
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d77143665bb1ea17cefea8d42d80003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61bd1e373f9845aca915c1b1d1456713
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91e7af10d4ee20f1e8f7e3a5df630b35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50398eaf8f3d9b9c285466c003f7bb78
    def get_inputs(self):
        return [
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c15f99175a74d6e38f2b36214f210aa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0808c9fcbc960a398d51addf53e19cab
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2561556861555eda4cf90cdba3362c65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa800e5f43222004df500bee2775470c
    def get_inputs(self):
        return [
            paddle.uniform([2054, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56223c566930865adafeaa1163b062bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30894a70778990ba804af814e417d17d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_59ef2c5a38d8108e8d38d4564c73499f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbe827cea03aaca01eb121df603bc8ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_8097e0da7dafcbcc57608c94c70e4ad9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f7a6a36e6d451e1241a2a9a5d7441b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a27bed94008e447e2933bcdf0a62f512(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_012768c0db613e5ecd733a6a0e0edd3f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.38251638412475586, 0.061437007039785385, 0.17679987847805023, 0.40426018834114075], [0.290205717086792, 0.4173891246318817, 0.03683020919561386, 0.13997584581375122], [0.41734710335731506, 0.4970444142818451, 0.04106966406106949, 0.2228603959083557], [0.41734710335731506, 0.4970444142818451, 0.04106966406106949, 0.2228603959083557], [0.4155576825141907, 0.17127370834350586, 0.3756653964519501, 0.12106780707836151]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.37363123893737793, 0.11039803177118301, 0.12842200696468353, 0.43897849321365356], [0.29999393224716187, 0.3632563650608063, 0.3421824276447296, 0.06522335857152939], [0.025168241932988167, 0.289311021566391, 0.23700326681137085, 0.16842719912528992], [0.025168241932988167, 0.289311021566391, 0.23700326681137085, 0.16842719912528992], [0.3235059380531311, 0.3044372498989105, 0.4845093786716461, 0.439764142036438]], dtype='float32').reshape([5, 4]),
        ]


class TestPrimitiveOp_79eac68b645986b64b21cf7bb9806364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79eac68b645986b64b21cf7bb9806364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79eac68b645986b64b21cf7bb9806364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79eac68b645986b64b21cf7bb9806364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79eac68b645986b64b21cf7bb9806364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79eac68b645986b64b21cf7bb9806364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79eac68b645986b64b21cf7bb9806364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366448733c73fe5def0cc7e7e8ec98a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5af25c119bbd17e3d9b11e40dc6757b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5af25c119bbd17e3d9b11e40dc6757b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5af25c119bbd17e3d9b11e40dc6757b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5af25c119bbd17e3d9b11e40dc6757b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5af25c119bbd17e3d9b11e40dc6757b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5af25c119bbd17e3d9b11e40dc6757b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5af25c119bbd17e3d9b11e40dc6757b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bfb56cd5191670ae5593396df788d44
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_a63b313cb0bce18c992ff08a2d9d9b7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bbd40843ed8df6da0b1fa7f67111020
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_91d7d6bf8395529118447431b20d1442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d7d6bf8395529118447431b20d1442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d7d6bf8395529118447431b20d1442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d7d6bf8395529118447431b20d1442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d7d6bf8395529118447431b20d1442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d7d6bf8395529118447431b20d1442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d7d6bf8395529118447431b20d1442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7b7fce290a11831e393e0b0a28619d13(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4218, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[4218, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e88d52829a31ba85e2a51678fe9b944b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b7fce290a11831e393e0b0a28619d13
    def get_inputs(self):
        return [
            paddle.uniform([4218, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_50462c25b335bee7210cf96c0e7f87ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4218, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4218, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_853aa264a9d32713cab17af471ad15e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50462c25b335bee7210cf96c0e7f87ac
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_853aa264a9d32713cab17af471ad15e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50462c25b335bee7210cf96c0e7f87ac
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_853aa264a9d32713cab17af471ad15e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50462c25b335bee7210cf96c0e7f87ac
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_853aa264a9d32713cab17af471ad15e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50462c25b335bee7210cf96c0e7f87ac
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_853aa264a9d32713cab17af471ad15e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50462c25b335bee7210cf96c0e7f87ac
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_853aa264a9d32713cab17af471ad15e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50462c25b335bee7210cf96c0e7f87ac
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_853aa264a9d32713cab17af471ad15e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50462c25b335bee7210cf96c0e7f87ac
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_853aa264a9d32713cab17af471ad15e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50462c25b335bee7210cf96c0e7f87ac
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_853aa264a9d32713cab17af471ad15e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50462c25b335bee7210cf96c0e7f87ac
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_853aa264a9d32713cab17af471ad15e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50462c25b335bee7210cf96c0e7f87ac
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_853aa264a9d32713cab17af471ad15e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50462c25b335bee7210cf96c0e7f87ac
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_897492c2c36080ae8e2d40d57ceee00c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_810dd930926bb2280bc25735c998c35b
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_e32bc5508569ce43601cbb337d0b6db9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12441866cc2f113c2daef98e28c90533
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e88d52829a31ba85e2a51678fe9b944b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b7fce290a11831e393e0b0a28619d13
    def get_inputs(self):
        return [
            paddle.uniform([4218, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0f583516e3459df5e49d7eca990bf55a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2f2b4663ad0e38d2e5e931474ae84ac
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3729301989078522, 0.4398423433303833, 0.13614453375339508, 0.014447365887463093], [0.15907423198223114, 0.3312903642654419, 0.45745378732681274, 0.44900935888290405], [0.2963332533836365, 0.3486698567867279, 0.47473400831222534, 0.3271164000034332], [0.3729301989078522, 0.4398423433303833, 0.13614453375339508, 0.014447365887463093], [0.26238396763801575, 0.021220184862613678, 0.11405127495527267, 0.05317042022943497], [0.30740490555763245, 0.23132562637329102, 0.13657315075397491, 0.21084262430667877], [0.26238396763801575, 0.021220184862613678, 0.11405127495527267, 0.05317042022943497]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.35818153619766235, 0.19112035632133484, 0.07843698561191559, 0.2016972452402115], [0.06737262010574341, 0.37645670771598816, 0.45104628801345825, 0.3832723796367645], [0.0901469737291336, 0.028215333819389343, 0.49505615234375, 0.394654244184494], [0.35818153619766235, 0.19112035632133484, 0.07843698561191559, 0.2016972452402115], [0.013244782574474812, 0.24561317265033722, 0.472744345664978, 0.11017251014709473], [0.2903924882411957, 0.2893523573875427, 0.09850277006626129, 0.10778801143169403], [0.013244782574474812, 0.24561317265033722, 0.472744345664978, 0.11017251014709473]], dtype='float32').reshape([7, 4]),
        ]


class TestPrimitiveOp_91d7d6bf8395529118447431b20d1442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d7d6bf8395529118447431b20d1442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d7d6bf8395529118447431b20d1442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d7d6bf8395529118447431b20d1442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d7d6bf8395529118447431b20d1442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d7d6bf8395529118447431b20d1442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d7d6bf8395529118447431b20d1442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b7bf90f773aa50024593645a0952115
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_6b6cc79803402b6b72e0b9e4edc79fde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86636cd620188e77aa5919cbf8b9d393
    def get_inputs(self):
        return [
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()