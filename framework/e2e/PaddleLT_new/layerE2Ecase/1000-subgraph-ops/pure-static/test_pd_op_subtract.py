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


class TestPrimitiveOp_ac5a478fb83caafc31e675dd17ac103c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dfcedf316eacbbadf7dd6cacaded896
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.39964500069618225]], [[0.12203215807676315]], [[0.010801950469613075]], [[0.3904440701007843]], [[0.1819305568933487]], [[0.12119834125041962]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.5297248363494873]], [[0.6716182827949524]], [[0.8010523915290833]], [[0.5264007449150085]], [[0.5280144810676575]], [[0.6123347282409668]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_781c77b21abb7a1af32d37b590f65ee1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dfcedf316eacbbadf7dd6cacaded896
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.30786585807800293]], [[0.18892748653888702]], [[0.2334328591823578]], [[0.39624476432800293]], [[0.007493613287806511]], [[0.26508504152297974]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.5312067866325378]], [[0.7162238359451294]], [[0.6836994290351868]], [[0.7592155337333679]], [[0.6527315378189087]], [[0.6262604594230652]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_b6de65a44204e6aecbe1290b1e2eaaf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2530a4e4625af688f4e3de9927836b64
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.21529504656791687, 0.1618257611989975]], [[0.009923112578690052, 0.34905481338500977]], [[0.06091126427054405, 0.32110556960105896]], [[0.11437404155731201, 0.359421968460083]], [[0.0612526573240757, 0.039813797920942307]], [[0.28820860385894775, 0.39143437147140503]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.22702056169509888, 0.1497810333967209]], [[0.20459942519664764, 0.1697428673505783]], [[0.2997417747974396, 0.16197894513607025]], [[0.15711261332035065, 0.16295047104358673]], [[0.14793767035007477, 0.31142574548721313]], [[0.3406373858451843, 0.09496405720710754]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_4f6cb79871f68b49894367d0cc3a0baa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2530a4e4625af688f4e3de9927836b64
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.2919347584247589, 0.3839297592639923]], [[0.12366976588964462, 0.016277754679322243]], [[0.3423648178577423, 0.3106607496738434]], [[0.48790884017944336, 0.33891406655311584]], [[0.06598566472530365, 0.05130679905414581]], [[0.18500781059265137, 0.2764788269996643]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.22702056169509888, 0.1497810333967209]], [[0.20459942519664764, 0.1697428673505783]], [[0.2997417747974396, 0.16197894513607025]], [[0.15711261332035065, 0.16295047104358673]], [[0.14793767035007477, 0.31142574548721313]], [[0.3406373858451843, 0.09496405720710754]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class TestPrimitiveOp_7accaf387a7161ce0013c96538dccc15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38e60ad0c3265f689e713b6b70f444bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.46265488862991333, 0.016697153449058533]], [[0.43998315930366516, 0.12417177855968475]], [[0.2822396457195282, 0.03765062242746353]], [[0.15949825942516327, 0.06882414221763611]], [[0.10706356912851334, 0.3038007915019989]], [[0.2781323492527008, 0.09815538674592972]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class TestPrimitiveOp_d2106f567fbe1f0d36b71e9802b028a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0368fad43623b73e790d949bddeb3e4a
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
            paddle.to_tensor([0.45525065064430237, 0.027886996045708656, 0.05469066649675369, 0.09922953695058823, 0.43944913148880005, 0.14764128625392914, 0.2596937119960785, 0.0014553589280694723, 0.2519356906414032, 0.26656362414360046, 0.04113401472568512, 0.14312632381916046, 0.42150378227233887, 0.17673715949058533, 0.0016911025159060955, 0.059259019792079926], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_b0e0c594aecd03d0400b43f921590069(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0368fad43623b73e790d949bddeb3e4a
    def get_inputs(self):
        return [
            paddle.to_tensor([0.45525065064430237, 0.027886996045708656, 0.05469066649675369, 0.09922953695058823, 0.43944913148880005, 0.14764128625392914, 0.2596937119960785, 0.0014553589280694723, 0.2519356906414032, 0.26656362414360046, 0.04113401472568512, 0.14312632381916046, 0.42150378227233887, 0.17673715949058533, 0.0016911025159060955, 0.059259019792079926], dtype='float32').reshape([16]),
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


class PrimitiveOp_420e0eff51951d42131471f99c7aa2af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1787, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1787, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9bbded82b5970e05ef17c7a4a3ea508f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_420e0eff51951d42131471f99c7aa2af
    def get_inputs(self):
        return [
            paddle.uniform([1787, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_382da77207b48ce623a0bb802b4c844e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1787, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1787, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_06ba3cf42a3ba5bf2c473e3b991d7536(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382da77207b48ce623a0bb802b4c844e
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06ba3cf42a3ba5bf2c473e3b991d7536(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382da77207b48ce623a0bb802b4c844e
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06ba3cf42a3ba5bf2c473e3b991d7536(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382da77207b48ce623a0bb802b4c844e
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06ba3cf42a3ba5bf2c473e3b991d7536(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382da77207b48ce623a0bb802b4c844e
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06ba3cf42a3ba5bf2c473e3b991d7536(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382da77207b48ce623a0bb802b4c844e
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06ba3cf42a3ba5bf2c473e3b991d7536(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382da77207b48ce623a0bb802b4c844e
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06ba3cf42a3ba5bf2c473e3b991d7536(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382da77207b48ce623a0bb802b4c844e
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06ba3cf42a3ba5bf2c473e3b991d7536(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382da77207b48ce623a0bb802b4c844e
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06ba3cf42a3ba5bf2c473e3b991d7536(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382da77207b48ce623a0bb802b4c844e
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06ba3cf42a3ba5bf2c473e3b991d7536(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382da77207b48ce623a0bb802b4c844e
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06ba3cf42a3ba5bf2c473e3b991d7536(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382da77207b48ce623a0bb802b4c844e
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_9bbded82b5970e05ef17c7a4a3ea508f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_420e0eff51951d42131471f99c7aa2af
    def get_inputs(self):
        return [
            paddle.uniform([1787, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1787, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_64d3629338b526228d9dc3dcdee17d33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_012768c0db613e5ecd733a6a0e0edd3f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3378430902957916, 0.29151809215545654, 0.0677536353468895, 0.4517274498939514], [0.058542750775814056, 0.4407898187637329, 0.2741307318210602, 0.4350442588329315], [0.18247398734092712, 0.1590757966041565, 0.047330133616924286, 0.4494421184062958], [0.4015616476535797, 0.19283300638198853, 0.09611863642930984, 0.3176335394382477], [0.2676127851009369, 0.04183512553572655, 0.22073909640312195, 0.2983342111110687]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.3270646929740906, 0.0601244755089283, 0.43532902002334595, 0.2411079704761505], [0.27831077575683594, 0.21018525958061218, 0.1494518667459488, 0.4174284040927887], [0.46858373284339905, 0.4413186311721802, 0.3147798180580139, 0.0042846654541790485], [0.3470216393470764, 0.03412919119000435, 0.11428213119506836, 0.32941684126853943], [0.43657946586608887, 0.4981544315814972, 0.10258132964372635, 0.032984960824251175]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_683d839a4d3f965162f03da002f26189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_012768c0db613e5ecd733a6a0e0edd3f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09803753346204758, 0.3370109796524048, 0.4920312464237213, 0.22063834965229034], [0.2076125591993332, 0.1940884292125702, 0.13235266506671906, 0.13326548039913177], [0.0006360704428516328, 0.4204197824001312, 0.2991025745868683, 0.3368101119995117], [0.2076125591993332, 0.1940884292125702, 0.13235266506671906, 0.13326548039913177], [0.0006360704428516328, 0.4204197824001312, 0.2991025745868683, 0.3368101119995117]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.06035241484642029, 0.022834384813904762, 0.39817655086517334, 0.03385337069630623], [0.1251024454832077, 0.12824144959449768, 0.24735508859157562, 0.30923792719841003], [0.2161232978105545, 0.2657049894332886, 0.0354897566139698, 0.39884719252586365], [0.1251024454832077, 0.12824144959449768, 0.24735508859157562, 0.30923792719841003], [0.2161232978105545, 0.2657049894332886, 0.0354897566139698, 0.39884719252586365]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_e36500f9f02c13a2b64748711369c4bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3243171274662018], [0.4460776746273041], [0.2511337995529175], [0.2209128737449646], [0.23234640061855316], [0.019881412386894226], [0.1990540474653244], [0.06053031235933304], [0.40715867280960083]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.340059757232666], [0.387004017829895], [0.49885281920433044], [0.4898195266723633], [0.3837003707885742], [0.4277181327342987], [0.0343017652630806], [0.48448464274406433], [0.25622862577438354]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_8ba694bf42cd33b3bf82a927e87b7469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3726502060890198], [0.0487419031560421], [0.026435574516654015], [0.3332546353340149], [0.025723274797201157], [0.2000621259212494], [0.2115304172039032], [0.06198172643780708], [0.0746815949678421]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.29957494139671326], [0.36993101239204407], [0.43416088819503784], [0.3709265887737274], [0.33858680725097656], [0.4856625199317932], [0.4982873797416687], [0.24978111684322357], [0.1863652914762497]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_5e7efbac3a018b2a3f8fabded2d38ee4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39222896099090576], [0.4460776746273041], [0.2511337995529175], [0.4043271541595459], [0.4428388476371765], [0.019881412386894226], [0.4657576084136963], [0.4012521505355835], [0.40715867280960083]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.340059757232666], [0.387004017829895], [0.4251270294189453], [0.4898195266723633], [0.31481853127479553], [0.4277181327342987], [0.0343017652630806], [0.3381534516811371], [0.09708041697740555]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_99cca1fabb2e799571a787b47c38f372(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3726502060890198], [0.35267090797424316], [0.026435574516654015], [0.40623462200164795], [0.3046559691429138], [0.4644503593444824], [0.2115304172039032], [0.4490237534046173], [0.2872993052005768]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.29957494139671326], [0.36993101239204407], [0.43416088819503784], [0.06923043727874756], [0.01820189692080021], [0.030403364449739456], [0.3196682929992676], [0.24978111684322357], [0.1863652914762497]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_d9c7a2f23f73bb4037342ffc8946c06c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3243171274662018], [0.49249011278152466], [0.2755092680454254], [0.2209128737449646], [0.23234640061855316], [0.21800071001052856], [0.1990540474653244], [0.06053031235933304], [0.49496304988861084]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.14785392582416534], [0.2970240116119385], [0.49885281920433044], [0.40843749046325684], [0.3837003707885742], [0.23476886749267578], [0.007210989482700825], [0.48448464274406433], [0.25622862577438354]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_7569e687f8e731c7986547ce47b56ca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.45809561014175415], [0.0487419031560421], [0.27907347679138184], [0.3332546353340149], [0.025723274797201157], [0.2000621259212494], [0.2891344726085663], [0.06198172643780708], [0.0746815949678421]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.08313236385583878], [0.07048333436250687], [0.31889981031417847], [0.3709265887737274], [0.33858680725097656], [0.4856625199317932], [0.4982873797416687], [0.249019056558609], [0.03643830493092537]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_177b30c1b5a302a8ade21155ac3d15ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06997948884963989], [-0.005269330460578203], [0.0798363983631134], [-0.021746868267655373], [0.08402508497238159], [-0.17223131656646729], [-0.08678125590085983], [0.09186724573373795], [0.04042743518948555]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_3b04136b2339fd402bae2966711f175f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39222896099090576], [0.49249011278152466], [0.2755092680454254], [0.4043271541595459], [0.4428388476371765], [0.21800071001052856], [0.4657576084136963], [0.4012521505355835], [0.49496304988861084]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.14785392582416534], [0.2970240116119385], [0.4251270294189453], [0.40843749046325684], [0.31481853127479553], [0.23476886749267578], [0.007210989482700825], [0.3381534516811371], [0.09708041697740555]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_4f8b05aa52de79ac0b1447d8ab4534b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.45809561014175415], [0.35267090797424316], [0.27907347679138184], [0.40623462200164795], [0.3046559691429138], [0.4644503593444824], [0.2891344726085663], [0.4490237534046173], [0.2872993052005768]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.08313236385583878], [0.07048333436250687], [0.31889981031417847], [0.06923043727874756], [0.01820189692080021], [0.030403364449739456], [0.3196682929992676], [0.249019056558609], [0.03643830493092537]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_35422b525f812c443bf2d124c49e0d77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09163165837526321], [0.0551581047475338], [0.005958727095276117], [-0.0013852004194632173], [0.036671943962574005], [-0.007278168108314276], [-0.014001179486513138], [0.012620036490261555], [0.09981323033571243]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.06997948884963989], [-0.005269330460578203], [0.0798363983631134], [-0.021746868267655373], [0.08402508497238159], [-0.17223131656646729], [-0.08678125590085983], [0.09186724573373795], [0.04042743518948555]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_2e291169f317ccdcf1b78c86733429e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [-0.0], [0.0], [-0.0], [0.0], [-0.0], [-0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.23629572987556458], [1.0955313444137573], [-12.39823055267334], [-14.699438095092773], [-1.291263461112976], [-22.66410255432129], [-5.19813871383667], [-6.279475212097168], [0.5949691534042358]], dtype='float32').reshape([9, 1]),
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


class TestPrimitiveOp_fc2f884cb10968c9ea915e1df3e3b856(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dfcedf316eacbbadf7dd6cacaded896
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3402940034866333]], [[0.3119332790374756]], [[0.03234013915061951]], [[0.06092549115419388]], [[0.23291897773742676]], [[0.03322348743677139]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6197690963745117]], [[0.80384761095047]], [[0.5083078145980835]], [[0.5051376819610596]], [[0.788497269153595]], [[0.5435491800308228]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_0f209224f57f4d66b20bdf8b82ddbdea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dfcedf316eacbbadf7dd6cacaded896
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.20923791825771332]], [[0.21416248381137848]], [[0.3052079975605011]], [[0.1252378523349762]], [[0.20577526092529297]], [[0.42120617628097534]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6410043835639954]], [[0.7822772264480591]], [[0.6408963203430176]], [[0.7869121432304382]], [[0.7918685674667358]], [[0.7693722248077393]]], dtype='float32').reshape([6, 1, 1]),
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


class PrimitiveOp_3fe52c635d00877404350c3e507c805b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5585, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[5585, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8abab98eefbfb66658535dc001a4072f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3fe52c635d00877404350c3e507c805b
    def get_inputs(self):
        return [
            paddle.uniform([5585, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9f2a15896a8f03d4ce31b312c6f3806b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5585, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5585, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_10ba31cc8cce81f0d4a28ec9f6c859b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f2a15896a8f03d4ce31b312c6f3806b
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10ba31cc8cce81f0d4a28ec9f6c859b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f2a15896a8f03d4ce31b312c6f3806b
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10ba31cc8cce81f0d4a28ec9f6c859b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f2a15896a8f03d4ce31b312c6f3806b
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10ba31cc8cce81f0d4a28ec9f6c859b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f2a15896a8f03d4ce31b312c6f3806b
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10ba31cc8cce81f0d4a28ec9f6c859b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f2a15896a8f03d4ce31b312c6f3806b
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10ba31cc8cce81f0d4a28ec9f6c859b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f2a15896a8f03d4ce31b312c6f3806b
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10ba31cc8cce81f0d4a28ec9f6c859b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f2a15896a8f03d4ce31b312c6f3806b
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10ba31cc8cce81f0d4a28ec9f6c859b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f2a15896a8f03d4ce31b312c6f3806b
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10ba31cc8cce81f0d4a28ec9f6c859b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f2a15896a8f03d4ce31b312c6f3806b
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10ba31cc8cce81f0d4a28ec9f6c859b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f2a15896a8f03d4ce31b312c6f3806b
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10ba31cc8cce81f0d4a28ec9f6c859b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f2a15896a8f03d4ce31b312c6f3806b
    def get_inputs(self):
        return [
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_8abab98eefbfb66658535dc001a4072f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3fe52c635d00877404350c3e507c805b
    def get_inputs(self):
        return [
            paddle.uniform([5585, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5585, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_0ecaca5e138b5390ff6bd2de9692916d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2f2b4663ad0e38d2e5e931474ae84ac
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3846586346626282, 0.2318805456161499, 0.2973344326019287, 0.09845169633626938], [0.40783971548080444, 0.12320192903280258, 0.44894203543663025, 0.08413249254226685], [0.008666963316500187, 0.3788311779499054, 0.23930314183235168, 0.2641450762748718], [0.40783971548080444, 0.12320192903280258, 0.44894203543663025, 0.08413249254226685], [0.008666963316500187, 0.3788311779499054, 0.23930314183235168, 0.2641450762748718], [0.3639656603336334, 0.09334275126457214, 0.2515942454338074, 0.07129544019699097], [0.3639656603336334, 0.09334275126457214, 0.2515942454338074, 0.07129544019699097]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.44040143489837646, 0.4227806627750397, 0.21807213127613068, 0.01137443259358406], [0.4793195128440857, 0.3249780833721161, 0.4924488365650177, 0.4538346230983734], [0.31182751059532166, 0.29544615745544434, 0.48342621326446533, 0.0304309893399477], [0.4793195128440857, 0.3249780833721161, 0.4924488365650177, 0.4538346230983734], [0.31182751059532166, 0.29544615745544434, 0.48342621326446533, 0.0304309893399477], [0.4627205729484558, 0.279140442609787, 0.47712820768356323, 0.2834685742855072], [0.4627205729484558, 0.279140442609787, 0.47712820768356323, 0.2834685742855072]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_875fff46184591cbb08eae0083270bc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1827273815870285, 0.4455740749835968, 0.4816020429134369, 0.39720895886421204, 0.35945671796798706, 0.4984632730484009], dtype='float32').reshape([6]),
            paddle.to_tensor([0.446471244096756, 0.14341187477111816, 0.4889226257801056, 0.025714602321386337, 0.20853398740291595, 0.15274621546268463], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_39e99be533aa568d2e4781e02550e1fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.011204774491488934, 0.1125684455037117, 0.10057477653026581, 0.2895292341709137, 0.3072856068611145, 0.31992271542549133], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2992474138736725, 0.2878594994544983, 0.47377514839172363, 0.05680272728204727, 0.043566759675741196, 0.43330734968185425], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_94845fe61f9eca5c6140e5d10f1741f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21949312090873718, 0.0012134052813053131, 0.4766252934932709, 0.11636532098054886, 0.2342921495437622, 0.4063635468482971], dtype='float32').reshape([6]),
            paddle.to_tensor([0.22107045352458954, 0.4040623903274536, 0.46366026997566223, 0.043019529432058334, 0.13830581307411194, 0.4344382584095001], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ee93be3c836d2286244e2188491f1b88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.031546175479888916, 0.1145879477262497, 0.332019567489624, 0.13446305692195892, 0.41516128182411194, 0.1730593889951706], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3741452395915985, 0.11825495213270187, 0.027543269097805023, 0.20650611817836761, 0.1986602246761322, 0.02518189512193203], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ed47fbc81eccd7600142fa60169dff85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21949312090873718, 0.0012134052813053131, 0.4766252934932709, 0.11636532098054886, 0.2342921495437622, 0.4063635468482971], dtype='float32').reshape([6]),
            paddle.to_tensor([0.446471244096756, 0.4040623903274536, 0.4889226257801056, 0.043019529432058334, 0.20853398740291595, 0.4344382584095001], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_efbb31876d4c72e868931c1a8c6a4647(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.031546175479888916, 0.1145879477262497, 0.332019567489624, 0.13446305692195892, 0.3072856068611145, 0.1730593889951706], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3741452395915985, 0.2878594994544983, 0.47377514839172363, 0.20650611817836761, 0.1986602246761322, 0.43330734968185425], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_6d4b8499be037869b659b4ccbb6d64d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.446471244096756, 0.4455740749835968, 0.4889226257801056, 0.39720895886421204, 0.35945671796798706, 0.4984632730484009], dtype='float32').reshape([6]),
            paddle.to_tensor([0.446471244096756, 0.14341187477111816, 0.4889226257801056, 0.025714602321386337, 0.20853398740291595, 0.15274621546268463], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_7fe9d7534ad57c4a54bf8111649efc09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2992474138736725, 0.2878594994544983, 0.47377514839172363, 0.2895292341709137, 0.3072856068611145, 0.43330734968185425], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2992474138736725, 0.2878594994544983, 0.47377514839172363, 0.05680272728204727, 0.043566759675741196, 0.43330734968185425], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_e73370579409983fbdb935d450022991(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0005403927061706781, 0.0014772489666938782, 0.0039475420489907265, 0.08117253333330154, 0.06058230996131897, -0.004151618108153343], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, 0.0, 0.0, -0.0, 0.0027979901060462, 0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_60ae8ab1e42de6c97fac55e342b79da2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.31459930539131165, 0.2944929599761963, 0.48526233434677124, 0.21146178245544434, 0.2839953601360321, 0.32560473680496216], dtype='float32').reshape([6]),
            paddle.to_tensor([0.22028177976608276, 0.20263789594173431, 0.47014278173446655, 0.07969242334365845, 0.18629898130893707, 0.4204009175300598], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_7eb2ccf6188806007447f01ad9ed0027(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.15522609651088715, 0.2002139687538147, 0.2871749699115753, 0.17316597700119019, 0.175426185131073, 0.376615047454834], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2028457075357437, 0.11642144620418549, 0.17978142201900482, 0.17048458755016327, 0.30691075325012207, 0.09912063926458359], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_363b8a6288b08b090cc0b00b94e19182(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.446471244096756, 0.4455740749835968, 0.4889226257801056, 0.39720895886421204, 0.35945671796798706, 0.4984632730484009], dtype='float32').reshape([6]),
            paddle.to_tensor([0.22107045352458954, 0.14341187477111816, 0.46366026997566223, 0.025714602321386337, 0.13830581307411194, 0.15274621546268463], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_27c6a5889a0d64a4b7c77a09465d12b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2992474138736725, 0.2878594994544983, 0.47377514839172363, 0.2895292341709137, 0.41516128182411194, 0.43330734968185425], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2992474138736725, 0.11825495213270187, 0.027543269097805023, 0.05680272728204727, 0.043566759675741196, 0.02518189512193203], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_1d59c016438d4765ff24cc094e42eee4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.004603986628353596, 1.5616939067840576, 0.042555682361125946, -0.7943583726882935, 0.4173123240470886, -0.18761827051639557], dtype='float32').reshape([6]),
            paddle.to_tensor([0.74139004945755, -1.0451209545135498, 0.019613176584243774, 1.0111474990844727, 0.5197926163673401, -1.2538810968399048], dtype='float32').reshape([6]),
        ]


class PrimitiveOp_2fe5c424b37447da94543750d4d8abf5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1774, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1774, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_11b4310df5bd9e4a48b2c945b4105173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fe5c424b37447da94543750d4d8abf5
    def get_inputs(self):
        return [
            paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c8815597bf87c32a16ed307a2028b1e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1774, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1774, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_449e33e541791edfdd659181ade9586d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8815597bf87c32a16ed307a2028b1e4
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_449e33e541791edfdd659181ade9586d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8815597bf87c32a16ed307a2028b1e4
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_449e33e541791edfdd659181ade9586d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8815597bf87c32a16ed307a2028b1e4
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_449e33e541791edfdd659181ade9586d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8815597bf87c32a16ed307a2028b1e4
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_449e33e541791edfdd659181ade9586d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8815597bf87c32a16ed307a2028b1e4
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_449e33e541791edfdd659181ade9586d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8815597bf87c32a16ed307a2028b1e4
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_449e33e541791edfdd659181ade9586d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8815597bf87c32a16ed307a2028b1e4
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_449e33e541791edfdd659181ade9586d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8815597bf87c32a16ed307a2028b1e4
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_449e33e541791edfdd659181ade9586d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8815597bf87c32a16ed307a2028b1e4
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_449e33e541791edfdd659181ade9586d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8815597bf87c32a16ed307a2028b1e4
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_449e33e541791edfdd659181ade9586d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8815597bf87c32a16ed307a2028b1e4
    def get_inputs(self):
        return [
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_11b4310df5bd9e4a48b2c945b4105173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fe5c424b37447da94543750d4d8abf5
    def get_inputs(self):
        return [
            paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_bdb79d67275ed4ddc8c8fde112b350a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0df867a4644c834ec9fd270790bdf2c
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4237799346446991, 0.11541059613227844, 0.00809119176119566, 0.3815240263938904, 0.059733111411333084, 0.08381999284029007, 0.3550366163253784, 0.2367265820503235, 0.3807695508003235, 0.32798030972480774, 0.31185442209243774, 0.3005569875240326, 0.4866105914115906, 0.01647285930812359, 0.10545752942562103, 0.422288715839386, 0.1458156257867813, 0.02166130021214485, 0.21083933115005493, 0.0744839757680893, 0.0265908632427454, 0.24041879177093506, 0.38260388374328613, 0.10905811190605164], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_9d00fdadf21a4d6f6a29b49a4e508272(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0df867a4644c834ec9fd270790bdf2c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4237799346446991, 0.11541059613227844, 0.00809119176119566, 0.3815240263938904, 0.059733111411333084, 0.08381999284029007, 0.3550366163253784, 0.2367265820503235, 0.3807695508003235, 0.32798030972480774, 0.31185442209243774, 0.3005569875240326, 0.4866105914115906, 0.01647285930812359, 0.10545752942562103, 0.422288715839386, 0.1458156257867813, 0.02166130021214485, 0.21083933115005493, 0.0744839757680893, 0.0265908632427454, 0.24041879177093506, 0.38260388374328613, 0.10905811190605164], dtype='float32').reshape([24]),
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


class PrimitiveOp_37c43a52c6e2d5f859660a3e574ef010(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1501, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1501, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ccf7a465307f5051fef6f7a78d12577(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37c43a52c6e2d5f859660a3e574ef010
    def get_inputs(self):
        return [
            paddle.uniform([1501, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_160d572e2635dcd11ec0ed456aed7cf4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1501, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1501, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a42e16a5674b52c4c6ab9d3aed2ba0b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_160d572e2635dcd11ec0ed456aed7cf4
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a42e16a5674b52c4c6ab9d3aed2ba0b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_160d572e2635dcd11ec0ed456aed7cf4
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a42e16a5674b52c4c6ab9d3aed2ba0b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_160d572e2635dcd11ec0ed456aed7cf4
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a42e16a5674b52c4c6ab9d3aed2ba0b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_160d572e2635dcd11ec0ed456aed7cf4
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a42e16a5674b52c4c6ab9d3aed2ba0b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_160d572e2635dcd11ec0ed456aed7cf4
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a42e16a5674b52c4c6ab9d3aed2ba0b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_160d572e2635dcd11ec0ed456aed7cf4
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a42e16a5674b52c4c6ab9d3aed2ba0b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_160d572e2635dcd11ec0ed456aed7cf4
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a42e16a5674b52c4c6ab9d3aed2ba0b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_160d572e2635dcd11ec0ed456aed7cf4
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a42e16a5674b52c4c6ab9d3aed2ba0b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_160d572e2635dcd11ec0ed456aed7cf4
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a42e16a5674b52c4c6ab9d3aed2ba0b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_160d572e2635dcd11ec0ed456aed7cf4
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a42e16a5674b52c4c6ab9d3aed2ba0b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_160d572e2635dcd11ec0ed456aed7cf4
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_7ccf7a465307f5051fef6f7a78d12577(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37c43a52c6e2d5f859660a3e574ef010
    def get_inputs(self):
        return [
            paddle.uniform([1501, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_07daa738aa86bf8ae6015cb00f3f859a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_473768ad5406d82abfbdd94c8502a0d7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
            paddle.to_tensor([0.43235915899276733, 0.37341681122779846, 0.39400672912597656, 0.1910826563835144], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_d3ac937b3c9a512bf737e089ea14594a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_473768ad5406d82abfbdd94c8502a0d7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.43235915899276733, 0.37341681122779846, 0.39400672912597656, 0.1910826563835144], dtype='float32').reshape([4]),
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


class TestPrimitiveOp_37449581e1a0cf528b1e802cca2a0156(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93bf25610fd14d81a6b319a42245208a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.31855759024620056, 0.4072616994380951, 0.37459737062454224, 0.4989188015460968], [0.4506174623966217, 0.17764601111412048, 0.10083357244729996, 0.2326803356409073], [0.09764281660318375, 0.4835951626300812, 0.1393831968307495, 0.4900238513946533], [0.3739396631717682, 0.11619941145181656, 0.3204847574234009, 0.30614128708839417], [0.3739396631717682, 0.11619941145181656, 0.3204847574234009, 0.30614128708839417], [0.09764281660318375, 0.4835951626300812, 0.1393831968307495, 0.4900238513946533]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.12331487983465195, 0.29535967111587524, 0.1293366700410843, 0.44953304529190063], [0.2046278864145279, 0.40882742404937744, 0.4848690629005432, 0.2765803933143616], [0.05675765872001648, 0.3153390884399414, 0.4283989369869232, 0.30572816729545593], [0.4577358663082123, 0.44362208247184753, 0.3328509032726288, 0.4255516231060028], [0.4577358663082123, 0.44362208247184753, 0.3328509032726288, 0.4255516231060028], [0.05675765872001648, 0.3153390884399414, 0.4283989369869232, 0.30572816729545593]], dtype='float32').reshape([6, 4]),
        ]


class TestPrimitiveOp_1817ae4047ac14c69f5e5ae95de59a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_012768c0db613e5ecd733a6a0e0edd3f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3564521372318268, 0.004584297072142363, 0.033546894788742065, 0.10575713962316513], [0.4161040782928467, 0.3088168501853943, 0.05969264358282089, 0.09086617827415466], [0.21536050736904144, 0.04620793089270592, 0.10334403812885284, 0.41686341166496277], [0.14949198067188263, 0.389742374420166, 0.4435247778892517, 0.3677942156791687], [0.3564521372318268, 0.004584297072142363, 0.033546894788742065, 0.10575713962316513]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.3681332767009735, 0.43496179580688477, 0.4649468958377838, 0.01326198223978281], [0.4411890506744385, 0.19893421232700348, 0.18682271242141724, 0.129261314868927], [0.3156616985797882, 0.04416563734412193, 0.12817716598510742, 0.4135863482952118], [0.06509086489677429, 0.14305001497268677, 0.044390808790922165, 0.1667936146259308], [0.3681332767009735, 0.43496179580688477, 0.4649468958377838, 0.01326198223978281]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_92c8ffc97f522f9ba3605e2135eb238d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.021477889269590378]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3178286850452423]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_99a93fba527514b7a4993da74a31400e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3298352062702179]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.43161070346832275]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_73a7b0d4bc5206a48d3b2f146ef61938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.021477889269590378]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.28265708684921265]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_e149ca2e8d3940dc394c3fb9d2fe6b54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.36230334639549255]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.06415817886590958]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_31f8f88c3fa38b3cf9625833b82b33b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4371948540210724]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3178286850452423]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_99a93fba527514b7a4993da74a31400e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3298352062702179]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.43161070346832275]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_9b0acb1847cb7ea672fa9b3086e16c1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.09001787006855011]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_20dbb0bd7cc1e728bed1e021280dc102(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4371948540210724]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.28265708684921265]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_e149ca2e8d3940dc394c3fb9d2fe6b54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.36230334639549255]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.06415817886590958]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_e5206a314bd1a71305627921e99343d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.04607468843460083]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.09001787006855011]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_6b25672c23e623b24ea329b7b8b8a508(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[2.95373797416687]], dtype='float32').reshape([1, 1]),
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


class TestPrimitiveOp_4f702c1814c1b4fbc4da694c80c3e095(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33286547660827637], [0.028109095990657806], [0.14381171762943268], [0.08470499515533447], [0.0046716793440282345], [0.15646472573280334]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.4764886200428009], [0.4810973107814789], [0.48731809854507446], [0.20144562423229218], [0.40835464000701904], [0.47783151268959045]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_dac704df765dfd8656c9334b2eb21535(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.26473021507263184], [0.4301255941390991], [0.009996733628213406], [0.43156635761260986], [0.02951250597834587], [0.05900372192263603]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.29171887040138245], [0.3921818137168884], [0.2787969708442688], [0.39924129843711853], [0.09039394557476044], [0.48042625188827515]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_fe2e69abeaf104054d62abb0bd0b45ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33286547660827637], [0.028109095990657806], [0.37021663784980774], [0.08470499515533447], [0.0046716793440282345], [0.19173109531402588]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.35102102160453796], [0.4810973107814789], [0.48731809854507446], [0.059361398220062256], [0.40835464000701904], [0.47783151268959045]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_1acf15e0d6a4d2301424e052503e6795(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.26473021507263184], [0.4301255941390991], [0.009996733628213406], [0.43156635761260986], [0.38260313868522644], [0.05900372192263603]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.10640901327133179], [0.15210367739200592], [0.058579739183187485], [0.04452253505587578], [0.053740326315164566], [0.4524851441383362]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_476df585e0f82c84e866b4d4b099ac41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3928629457950592], [0.3117378354072571], [0.14381171762943268], [0.40953099727630615], [0.0882832407951355], [0.15646472573280334]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.4764886200428009], [0.21417561173439026], [0.3296174705028534], [0.20144562423229218], [0.09571299701929092], [0.09185384213924408]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_bd80a7860df63574ebf8064a70fe9920(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.43764728307724], [0.4305558502674103], [0.3785059452056885], [0.4988960921764374], [0.02951250597834587], [0.22250328958034515]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.29171887040138245], [0.3921818137168884], [0.2787969708442688], [0.39924129843711853], [0.09039394557476044], [0.48042625188827515]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_202472f592cb9b35e9ea6435efde22a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.015077769756317139], [-0.12219679355621338], [-0.012837361544370651], [0.030545789748430252], [-0.13230396807193756], [0.09591057151556015]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_bfaa794f5655cca20ad562856da0c7d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3928629457950592], [0.3117378354072571], [0.37021663784980774], [0.40953099727630615], [0.0882832407951355], [0.19173109531402588]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.35102102160453796], [0.21417561173439026], [0.3296174705028534], [0.059361398220062256], [0.09571299701929092], [0.09185384213924408]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_55e5f42ffe667bd09ef0c498208cbbab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.43764728307724], [0.4305558502674103], [0.3785059452056885], [0.4988960921764374], [0.38260313868522644], [0.22250328958034515]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.10640901327133179], [0.15210367739200592], [0.058579739183187485], [0.04452253505587578], [0.053740326315164566], [0.4524851441383362]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_b830712aa0b4bc8965d14a65905d3626(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.013859646394848824], [0.02716641128063202], [0.012988737784326077], [0.15910780429840088], [-0.0024433706421405077], [-0.022969955578446388]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.015077769756317139], [-0.12219679355621338], [-0.012837361544370651], [0.030545789748430252], [-0.13230396807193756], [0.09591057151556015]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_b2ed8358916da609800af3f6e194015f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [-0.0], [-0.0], [0.0], [-0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[2.0878899097442627], [5.498083591461182], [1.9883456230163574], [0.8080183267593384], [-53.148136138916016], [5.175478935241699]], dtype='float32').reshape([6, 1]),
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


class TestPrimitiveOp_d748a258a10f28a8b53269ca60ebf1b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e223e011a18f61a428db8a806582d14
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2605110704898834, 0.1201103925704956, 0.20254603028297424, 0.03316880762577057], [0.26343226432800293, 0.004121940582990646, 0.11496882140636444, 0.4061180353164673], [0.1584235429763794, 0.3239191770553589, 0.28159084916114807, 0.08888214081525803], [0.3513008654117584, 0.20510320365428925, 0.06991801410913467, 0.2686355412006378]], dtype='float32').reshape([4, 4]),
            paddle.to_tensor([[0.40678662061691284, 0.25126999616622925, 0.3455744683742523, 0.13415615260601044], [0.02980189025402069, 0.1353379189968109, 0.14692826569080353, 0.03482954576611519], [0.3668142259120941, 0.3141435980796814, 0.18701907992362976, 0.2654780149459839], [0.2105587273836136, 0.3588237762451172, 0.05105412378907204, 0.07095363736152649]], dtype='float32').reshape([4, 4]),
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


class PrimitiveOp_21bc04b9f3905a57d013654cf65e4dcf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2049, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2049, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74297f8d577b468604e3f6fb47fedef3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21bc04b9f3905a57d013654cf65e4dcf
    def get_inputs(self):
        return [
            paddle.uniform([2049, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ed093f7d5d751f68cef2c6f56e129748(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2049, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2049, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_acd55aa4911fa36c11c3d4e9cfdd8294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed093f7d5d751f68cef2c6f56e129748
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acd55aa4911fa36c11c3d4e9cfdd8294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed093f7d5d751f68cef2c6f56e129748
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acd55aa4911fa36c11c3d4e9cfdd8294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed093f7d5d751f68cef2c6f56e129748
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acd55aa4911fa36c11c3d4e9cfdd8294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed093f7d5d751f68cef2c6f56e129748
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acd55aa4911fa36c11c3d4e9cfdd8294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed093f7d5d751f68cef2c6f56e129748
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acd55aa4911fa36c11c3d4e9cfdd8294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed093f7d5d751f68cef2c6f56e129748
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acd55aa4911fa36c11c3d4e9cfdd8294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed093f7d5d751f68cef2c6f56e129748
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acd55aa4911fa36c11c3d4e9cfdd8294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed093f7d5d751f68cef2c6f56e129748
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acd55aa4911fa36c11c3d4e9cfdd8294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed093f7d5d751f68cef2c6f56e129748
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acd55aa4911fa36c11c3d4e9cfdd8294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed093f7d5d751f68cef2c6f56e129748
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acd55aa4911fa36c11c3d4e9cfdd8294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed093f7d5d751f68cef2c6f56e129748
    def get_inputs(self):
        return [
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_74297f8d577b468604e3f6fb47fedef3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21bc04b9f3905a57d013654cf65e4dcf
    def get_inputs(self):
        return [
            paddle.uniform([2049, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2049, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecf30ea1dbf2e7288041a9f49f6ae3f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2f2b4663ad0e38d2e5e931474ae84ac
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07113082706928253, 0.2983020544052124, 0.04840359091758728, 0.2039358764886856], [0.07113082706928253, 0.2983020544052124, 0.04840359091758728, 0.2039358764886856], [0.2788667678833008, 0.41633340716362, 0.16865180432796478, 0.16923803091049194], [0.41474679112434387, 0.1731586754322052, 0.1571117788553238, 0.49831047654151917], [0.18186615407466888, 0.46318936347961426, 0.19186224043369293, 0.1651674062013626], [0.045829251408576965, 0.33112651109695435, 0.04129958152770996, 0.28658097982406616], [0.2631915509700775, 0.1978447586297989, 0.22431278228759766, 0.07939860224723816]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.3472058176994324, 0.13025164604187012, 0.03228257969021797, 0.002678100485354662], [0.3472058176994324, 0.13025164604187012, 0.03228257969021797, 0.002678100485354662], [0.0010673562064766884, 0.2506611943244934, 0.24540400505065918, 0.02254461869597435], [0.006475923117250204, 0.0036094021052122116, 0.15065337717533112, 0.18355880677700043], [0.43969133496284485, 0.47572702169418335, 0.33448490500450134, 0.01949414797127247], [0.011924875900149345, 0.3017769753932953, 0.19757194817066193, 0.05662257596850395], [0.2938377857208252, 0.3796485364437103, 0.1587771326303482, 0.1014678105711937]], dtype='float32').reshape([7, 4]),
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


class PrimitiveOp_6c9c24076bc52dba5195cf317705c743(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4634, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[4634, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0af6854c73e77c263f8a0a87f531a366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c9c24076bc52dba5195cf317705c743
    def get_inputs(self):
        return [
            paddle.uniform([4634, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5f193c5355f8363def5eff78ff7d3c7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4634, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4634, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_58c10820409dbc5af8103a86e5166a4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f193c5355f8363def5eff78ff7d3c7d
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58c10820409dbc5af8103a86e5166a4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f193c5355f8363def5eff78ff7d3c7d
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58c10820409dbc5af8103a86e5166a4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f193c5355f8363def5eff78ff7d3c7d
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58c10820409dbc5af8103a86e5166a4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f193c5355f8363def5eff78ff7d3c7d
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58c10820409dbc5af8103a86e5166a4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f193c5355f8363def5eff78ff7d3c7d
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58c10820409dbc5af8103a86e5166a4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f193c5355f8363def5eff78ff7d3c7d
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58c10820409dbc5af8103a86e5166a4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f193c5355f8363def5eff78ff7d3c7d
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58c10820409dbc5af8103a86e5166a4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f193c5355f8363def5eff78ff7d3c7d
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58c10820409dbc5af8103a86e5166a4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f193c5355f8363def5eff78ff7d3c7d
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58c10820409dbc5af8103a86e5166a4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f193c5355f8363def5eff78ff7d3c7d
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58c10820409dbc5af8103a86e5166a4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f193c5355f8363def5eff78ff7d3c7d
    def get_inputs(self):
        return [
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_0af6854c73e77c263f8a0a87f531a366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c9c24076bc52dba5195cf317705c743
    def get_inputs(self):
        return [
            paddle.uniform([4634, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4634, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f43dc3d411fccc9145d8834938e56b7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1000, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1000, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_06180864249312604ada1ad28fdd7f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f43dc3d411fccc9145d8834938e56b7d
    def get_inputs(self):
        return [
            paddle.uniform([1000, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f5548b9dc2b73a2a368facecdc87719b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1000, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1000, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_174c532c54fd30455defd8f398910380(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5548b9dc2b73a2a368facecdc87719b
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_174c532c54fd30455defd8f398910380(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5548b9dc2b73a2a368facecdc87719b
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_174c532c54fd30455defd8f398910380(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5548b9dc2b73a2a368facecdc87719b
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_174c532c54fd30455defd8f398910380(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5548b9dc2b73a2a368facecdc87719b
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_174c532c54fd30455defd8f398910380(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5548b9dc2b73a2a368facecdc87719b
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_174c532c54fd30455defd8f398910380(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5548b9dc2b73a2a368facecdc87719b
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_174c532c54fd30455defd8f398910380(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5548b9dc2b73a2a368facecdc87719b
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_174c532c54fd30455defd8f398910380(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5548b9dc2b73a2a368facecdc87719b
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_174c532c54fd30455defd8f398910380(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5548b9dc2b73a2a368facecdc87719b
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_174c532c54fd30455defd8f398910380(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5548b9dc2b73a2a368facecdc87719b
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_174c532c54fd30455defd8f398910380(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5548b9dc2b73a2a368facecdc87719b
    def get_inputs(self):
        return [
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_06180864249312604ada1ad28fdd7f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f43dc3d411fccc9145d8834938e56b7d
    def get_inputs(self):
        return [
            paddle.uniform([1000, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_30ca161ee5dd8e4be29feee7e2b8835d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93bf25610fd14d81a6b319a42245208a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.43048322200775146, 0.07031414657831192, 0.24098792672157288, 0.333217591047287], [0.3312043249607086, 0.360426664352417, 0.05441081523895264, 0.07200492173433304], [0.3312043249607086, 0.360426664352417, 0.05441081523895264, 0.07200492173433304], [0.21512995660305023, 0.15553048253059387, 0.11219269037246704, 0.3693389594554901], [0.420112669467926, 0.4652494192123413, 0.4867617189884186, 0.20929241180419922], [0.026353809982538223, 0.060513533651828766, 0.2639797031879425, 0.04610634222626686]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.011094214394688606, 0.3088628053665161, 0.06188592314720154, 0.07780618220567703], [0.1764441728591919, 0.23255352675914764, 0.3284577429294586, 0.06442199647426605], [0.1764441728591919, 0.23255352675914764, 0.3284577429294586, 0.06442199647426605], [0.4087875485420227, 0.08858349174261093, 0.49698585271835327, 0.31775298714637756], [0.059702835977077484, 0.46418848633766174, 0.3791363537311554, 0.4984138011932373], [0.19469574093818665, 0.3906388282775879, 0.4922601580619812, 0.07688571512699127]], dtype='float32').reshape([6, 4]),
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


class TestPrimitiveOp_1b7c74b51bd27f65b17c6ee8389b9c53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1546d5060eb5e7f91c606a3154f7d7bd
    def get_inputs(self):
        return [
            paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.6367955207824707, 3.6142578125, 0.26159077882766724, 10.938944816589355], [0.37410667538642883, 3.685392379760742, 2.6679515838623047, 3.5313498973846436]], dtype='float32').reshape([2, 4]),
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


class TestPrimitiveOp_b3bc095753cb9b077a7aeebcafa754b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29eb514882c92ab3a86517239fc1edf9
    def get_inputs(self):
        return [
            paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.48000794649124146, 1.8229349851608276, 22.99870491027832, 26.345333099365234], [2.3090498447418213, 0.4250306189060211, 1.5782874822616577, 1.4373853206634521]], dtype='float32').reshape([2, 4]),
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


class TestPrimitiveOp_bff9ac8b9a25ae2195a4d0d8d867ef9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3062000572681427], [0.03344854712486267], [0.07517310976982117], [0.19642746448516846], [0.023015061393380165]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.24273258447647095], [0.08545062690973282], [0.35097381472587585], [0.4020307958126068], [0.36489900946617126]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_9a8414263c0cbbaf98fd6d210e85cd35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.02279740385711193], [0.15860792994499207], [0.21742399036884308], [0.2578918933868408], [0.05762705206871033]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.15814867615699768], [0.42982131242752075], [0.37481895089149475], [0.38867270946502686], [0.125055193901062]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_e74bbfa1ca294e6c1ff841ff0e68cbbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3062000572681427], [0.3130226135253906], [0.17877483367919922], [0.19642746448516846], [0.023015061393380165]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.10822069644927979], [0.06122198328375816], [0.35097381472587585], [0.4020307958126068], [0.36489900946617126]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_1c89a62e2486def3651622ab115c9914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.02279740385711193], [0.34154266119003296], [0.30135348439216614], [0.4762178957462311], [0.05762705206871033]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.1466745138168335], [0.2202141284942627], [0.20410498976707458], [0.19355495274066925], [0.125055193901062]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_a3306a6680f15ff7f339eb26dfadae96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.35707125067710876], [0.03344854712486267], [0.07517310976982117], [0.34104833006858826], [0.4732273817062378]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.24273258447647095], [0.08545062690973282], [0.11754946410655975], [0.3820875585079193], [0.32701370120048523]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_8ee2a65e81dd0130eb2a1d7eb1b5581e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20563507080078125], [0.15860792994499207], [0.21742399036884308], [0.2578918933868408], [0.23720066249370575]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.15814867615699768], [0.42982131242752075], [0.37481895089149475], [0.38867270946502686], [0.0035878224298357964]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_100d4dcea71b49641f466140ec973e2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.019095581024885178], [0.04465426132082939], [-0.010076267644762993], [-0.05274929478764534], [0.05720999091863632]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_cc0a03ecf94527af2aaf8d2115345e51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.35707125067710876], [0.3130226135253906], [0.17877483367919922], [0.34104833006858826], [0.4732273817062378]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.10822069644927979], [0.06122198328375816], [0.11754946410655975], [0.3820875585079193], [0.32701370120048523]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_512b2450b8d66c16762e702cde1d6a02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20563507080078125], [0.34154266119003296], [0.30135348439216614], [0.4762178957462311], [0.23720066249370575]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.1466745138168335], [0.2202141284942627], [0.20410498976707458], [0.19355495274066925], [0.0035878224298357964]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_dc1f76045183faecbe1d8b14745b5a39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.01467236690223217], [0.030550600960850716], [0.005954075139015913], [-0.011600268073379993], [0.034157391637563705]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.019095581024885178], [0.04465426132082939], [-0.010076267644762993], [-0.05274929478764534], [0.05720999091863632]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_f0d2c1246210582ebb51e0ac44995085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [0.0], [-0.0], [-0.0], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[2.3014655113220215], [-0.46164920926094055], [2.692331314086914], [-3.547247886657715], [-0.6748934388160706]], dtype='float32').reshape([5, 1]),
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


class PrimitiveOp_2f5bcba4623b26140f0d3a9eb3242e7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2382, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2382, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf573667e463a298af51a637eefe3b3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f5bcba4623b26140f0d3a9eb3242e7c
    def get_inputs(self):
        return [
            paddle.uniform([2382, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_430b321845f14418d93329bc437bae9c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2382, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2382, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc2eaf77abde62fcae4726677399640e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_430b321845f14418d93329bc437bae9c
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc2eaf77abde62fcae4726677399640e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_430b321845f14418d93329bc437bae9c
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc2eaf77abde62fcae4726677399640e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_430b321845f14418d93329bc437bae9c
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc2eaf77abde62fcae4726677399640e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_430b321845f14418d93329bc437bae9c
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc2eaf77abde62fcae4726677399640e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_430b321845f14418d93329bc437bae9c
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc2eaf77abde62fcae4726677399640e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_430b321845f14418d93329bc437bae9c
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc2eaf77abde62fcae4726677399640e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_430b321845f14418d93329bc437bae9c
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc2eaf77abde62fcae4726677399640e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_430b321845f14418d93329bc437bae9c
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc2eaf77abde62fcae4726677399640e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_430b321845f14418d93329bc437bae9c
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc2eaf77abde62fcae4726677399640e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_430b321845f14418d93329bc437bae9c
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc2eaf77abde62fcae4726677399640e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_430b321845f14418d93329bc437bae9c
    def get_inputs(self):
        return [
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_bf573667e463a298af51a637eefe3b3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f5bcba4623b26140f0d3a9eb3242e7c
    def get_inputs(self):
        return [
            paddle.uniform([2382, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2382, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a190b1f15a0c7e41a9177b332b67454b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2976, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2976, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a5dcb6c6cee746e21e9503ebed531a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a190b1f15a0c7e41a9177b332b67454b
    def get_inputs(self):
        return [
            paddle.uniform([2976, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5f520b872f9e54a3d7c676b9f436fe24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2976, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2976, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b397bf6ea303a224155b856da47412cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f520b872f9e54a3d7c676b9f436fe24
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b397bf6ea303a224155b856da47412cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f520b872f9e54a3d7c676b9f436fe24
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b397bf6ea303a224155b856da47412cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f520b872f9e54a3d7c676b9f436fe24
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b397bf6ea303a224155b856da47412cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f520b872f9e54a3d7c676b9f436fe24
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b397bf6ea303a224155b856da47412cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f520b872f9e54a3d7c676b9f436fe24
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b397bf6ea303a224155b856da47412cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f520b872f9e54a3d7c676b9f436fe24
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b397bf6ea303a224155b856da47412cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f520b872f9e54a3d7c676b9f436fe24
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b397bf6ea303a224155b856da47412cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f520b872f9e54a3d7c676b9f436fe24
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b397bf6ea303a224155b856da47412cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f520b872f9e54a3d7c676b9f436fe24
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b397bf6ea303a224155b856da47412cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f520b872f9e54a3d7c676b9f436fe24
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b397bf6ea303a224155b856da47412cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f520b872f9e54a3d7c676b9f436fe24
    def get_inputs(self):
        return [
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_3a5dcb6c6cee746e21e9503ebed531a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a190b1f15a0c7e41a9177b332b67454b
    def get_inputs(self):
        return [
            paddle.uniform([2976, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2976, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3949720512062ae97f03227f7b303ecb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3753, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[3753, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_909584ee4c24a9f8d93a7bcd2aa3a459(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3949720512062ae97f03227f7b303ecb
    def get_inputs(self):
        return [
            paddle.uniform([3753, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0158ceda674393843ca3db0437c760b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3753, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3753, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dd5dba5e29f9fa7e80c48d23d16ac216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0158ceda674393843ca3db0437c760b7
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd5dba5e29f9fa7e80c48d23d16ac216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0158ceda674393843ca3db0437c760b7
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd5dba5e29f9fa7e80c48d23d16ac216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0158ceda674393843ca3db0437c760b7
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd5dba5e29f9fa7e80c48d23d16ac216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0158ceda674393843ca3db0437c760b7
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd5dba5e29f9fa7e80c48d23d16ac216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0158ceda674393843ca3db0437c760b7
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd5dba5e29f9fa7e80c48d23d16ac216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0158ceda674393843ca3db0437c760b7
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd5dba5e29f9fa7e80c48d23d16ac216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0158ceda674393843ca3db0437c760b7
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd5dba5e29f9fa7e80c48d23d16ac216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0158ceda674393843ca3db0437c760b7
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd5dba5e29f9fa7e80c48d23d16ac216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0158ceda674393843ca3db0437c760b7
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd5dba5e29f9fa7e80c48d23d16ac216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0158ceda674393843ca3db0437c760b7
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd5dba5e29f9fa7e80c48d23d16ac216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0158ceda674393843ca3db0437c760b7
    def get_inputs(self):
        return [
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_909584ee4c24a9f8d93a7bcd2aa3a459(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3949720512062ae97f03227f7b303ecb
    def get_inputs(self):
        return [
            paddle.uniform([3753, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3753, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_b33997181eb953619195c4794c222521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_042c31aed200f7f6f007e22c52c71138
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
            paddle.to_tensor([0.4680095911026001, 0.41818323731422424, 0.14651672542095184, 0.280924916267395, 0.4328104853630066, 0.054005738347768784, 0.3026198148727417, 0.2460148185491562, 0.2609083652496338, 0.20063157379627228, 0.007730483077466488, 0.14513055980205536, 0.1574631929397583, 0.22134825587272644, 0.48197489976882935, 0.39677584171295166, 0.10996971279382706, 0.21447205543518066, 0.26940596103668213, 0.01980494149029255], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_01158d87bbc7775eb7499625a0052f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_042c31aed200f7f6f007e22c52c71138
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4680095911026001, 0.41818323731422424, 0.14651672542095184, 0.280924916267395, 0.4328104853630066, 0.054005738347768784, 0.3026198148727417, 0.2460148185491562, 0.2609083652496338, 0.20063157379627228, 0.007730483077466488, 0.14513055980205536, 0.1574631929397583, 0.22134825587272644, 0.48197489976882935, 0.39677584171295166, 0.10996971279382706, 0.21447205543518066, 0.26940596103668213, 0.01980494149029255], dtype='float32').reshape([20]),
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


class TestPrimitiveOp_1b3b9ce4cfa7a7154a6b4604773b0705(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.25048601627349854], [0.2475523203611374], [0.09424278885126114], [0.2765282988548279]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3934442102909088], [0.42687827348709106], [0.35247161984443665], [0.20445282757282257]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_597c7fa6a6b840378600c4d422a9aca6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1932741403579712], [0.030457817018032074], [0.12248531728982925], [0.3236304521560669]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4565404951572418], [0.4529203772544861], [0.4354965388774872], [0.31460875272750854]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_89ad250b3885bca34b5db7c2dfa43864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4963822662830353], [0.2475523203611374], [0.11298491805791855], [0.2765282988548279]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.17840133607387543], [0.15334481000900269], [0.35247161984443665], [0.10544773191213608]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_5aa35d8124b0ad309f891d5be640d43d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2910788357257843], [0.030457817018032074], [0.4478048086166382], [0.3236304521560669]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.42456135153770447], [0.41002196073532104], [0.4354965388774872], [0.12312145531177521]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_9cc49912600851d19148e2a1df0f09e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.25048601627349854], [0.43678945302963257], [0.09424278885126114], [0.2916932702064514]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3934442102909088], [0.42687827348709106], [0.24674589931964874], [0.20445282757282257]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_d8fdfae33f911bed707bfae1ce956c79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1932741403579712], [0.1818334460258484], [0.12248531728982925], [0.3949283957481384]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4565404951572418], [0.4529203772544861], [0.1775546818971634], [0.31460875272750854]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_01a5fda2f30e9021003aac029f7eeda1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0048088133335113525], [-0.038444582372903824], [0.005450582131743431], [0.04131031408905983]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0006502432515844703]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_00081be27ecbcd1237dec51b670ea9be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4963822662830353], [0.43678945302963257], [0.11298491805791855], [0.2916932702064514]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.17840133607387543], [0.15334481000900269], [0.24674589931964874], [0.10544773191213608]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_0bd06a00d2d1b6c67fa38b92949e92c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2910788357257843], [0.1818334460258484], [0.4478048086166382], [0.3949283957481384]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.42456135153770447], [0.41002196073532104], [0.1775546818971634], [0.12312145531177521]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_b51e20e7f92338d01e52ac0f753606e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.04244489595293999], [-0.06467881053686142], [-0.03614892438054085], [0.05062283203005791]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.0048088133335113525], [-0.038444582372903824], [0.005450582131743431], [0.0406600721180439]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_6d5fceddacf652443414cc433d14042f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [-0.0], [0.0], [0.01599218137562275]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.8867045640945435], [0.40560775995254517], [1.150781273841858], [0.19680368900299072]], dtype='float32').reshape([4, 1]),
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


class PrimitiveOp_bc142c41d9499e3eb760e1df7667952f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1995, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1995, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9e2649f2e050e1b0a2df0b483bada92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc142c41d9499e3eb760e1df7667952f
    def get_inputs(self):
        return [
            paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b71f18e04ca628ed260ab8cb38d9be21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1995, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1995, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a2661035aa42330da13b27f514624468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b71f18e04ca628ed260ab8cb38d9be21
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2661035aa42330da13b27f514624468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b71f18e04ca628ed260ab8cb38d9be21
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2661035aa42330da13b27f514624468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b71f18e04ca628ed260ab8cb38d9be21
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2661035aa42330da13b27f514624468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b71f18e04ca628ed260ab8cb38d9be21
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2661035aa42330da13b27f514624468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b71f18e04ca628ed260ab8cb38d9be21
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2661035aa42330da13b27f514624468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b71f18e04ca628ed260ab8cb38d9be21
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2661035aa42330da13b27f514624468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b71f18e04ca628ed260ab8cb38d9be21
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2661035aa42330da13b27f514624468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b71f18e04ca628ed260ab8cb38d9be21
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2661035aa42330da13b27f514624468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b71f18e04ca628ed260ab8cb38d9be21
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2661035aa42330da13b27f514624468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b71f18e04ca628ed260ab8cb38d9be21
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2661035aa42330da13b27f514624468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b71f18e04ca628ed260ab8cb38d9be21
    def get_inputs(self):
        return [
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f9e2649f2e050e1b0a2df0b483bada92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc142c41d9499e3eb760e1df7667952f
    def get_inputs(self):
        return [
            paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_8816a45efccab2f295ad0d51562023c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_012768c0db613e5ecd733a6a0e0edd3f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3456915616989136, 0.2506336271762848, 0.06624741852283478, 0.3710392117500305], [0.09271609783172607, 0.04946192726492882, 0.36885109543800354, 0.2925843894481659], [0.3517704904079437, 0.2751832604408264, 0.48225122690200806, 0.06572527438402176], [0.3517704904079437, 0.2751832604408264, 0.48225122690200806, 0.06572527438402176], [0.4481711685657501, 0.4714016616344452, 0.3636152744293213, 0.4887006878852844]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.2955966591835022, 0.09677700698375702, 0.3422030508518219, 0.04490967467427254], [0.20998303592205048, 0.16991610825061798, 0.3474442660808563, 0.4242284595966339], [0.3717971444129944, 0.1900642365217209, 0.21085576713085175, 0.12603788077831268], [0.3717971444129944, 0.1900642365217209, 0.21085576713085175, 0.12603788077831268], [0.10295776277780533, 0.42835360765457153, 0.4322146475315094, 0.24509505927562714]], dtype='float32').reshape([5, 4]),
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


class PrimitiveOp_6ee3a446548ba52037c8e79827a94784(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[4185, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_619cd3a209d93d920439b50a80c6b2f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee3a446548ba52037c8e79827a94784
    def get_inputs(self):
        return [
            paddle.uniform([4185, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0434092dd79a5708bf98aad33cb59802(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4185, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_05bf7d5133185aacf215c1372ccc28b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0434092dd79a5708bf98aad33cb59802
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05bf7d5133185aacf215c1372ccc28b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0434092dd79a5708bf98aad33cb59802
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05bf7d5133185aacf215c1372ccc28b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0434092dd79a5708bf98aad33cb59802
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05bf7d5133185aacf215c1372ccc28b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0434092dd79a5708bf98aad33cb59802
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05bf7d5133185aacf215c1372ccc28b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0434092dd79a5708bf98aad33cb59802
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05bf7d5133185aacf215c1372ccc28b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0434092dd79a5708bf98aad33cb59802
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05bf7d5133185aacf215c1372ccc28b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0434092dd79a5708bf98aad33cb59802
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05bf7d5133185aacf215c1372ccc28b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0434092dd79a5708bf98aad33cb59802
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05bf7d5133185aacf215c1372ccc28b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0434092dd79a5708bf98aad33cb59802
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05bf7d5133185aacf215c1372ccc28b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0434092dd79a5708bf98aad33cb59802
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05bf7d5133185aacf215c1372ccc28b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0434092dd79a5708bf98aad33cb59802
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_619cd3a209d93d920439b50a80c6b2f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ee3a446548ba52037c8e79827a94784
    def get_inputs(self):
        return [
            paddle.uniform([4185, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d49b280d3e5cc0a9442eac490e17e243(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2f2b4663ad0e38d2e5e931474ae84ac
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4066733121871948, 0.2793608009815216, 0.21632757782936096, 0.08478543907403946], [0.011855659075081348, 0.32745808362960815, 0.11983894556760788, 0.3634859025478363], [0.15219391882419586, 0.09388584643602371, 0.44230917096138, 0.15760618448257446], [0.4066733121871948, 0.2793608009815216, 0.21632757782936096, 0.08478543907403946], [0.45928651094436646, 0.4210803806781769, 0.03597502410411835, 0.2904563248157501], [0.48192349076271057, 0.22896654903888702, 0.3035208284854889, 0.33978262543678284], [0.45928651094436646, 0.4210803806781769, 0.03597502410411835, 0.2904563248157501]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.3035692572593689, 0.03863402456045151, 0.08272188901901245, 0.41632160544395447], [0.42597702145576477, 0.09890887886285782, 0.44311729073524475, 0.38154807686805725], [0.33421626687049866, 0.08432349562644958, 0.2855485677719116, 0.16561037302017212], [0.3035692572593689, 0.03863402456045151, 0.08272188901901245, 0.41632160544395447], [0.08075264096260071, 0.38523122668266296, 0.443812757730484, 0.16283686459064484], [0.3726046681404114, 0.015341859310865402, 0.1500067263841629, 0.425263375043869], [0.08075264096260071, 0.38523122668266296, 0.443812757730484, 0.16283686459064484]], dtype='float32').reshape([7, 4]),
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