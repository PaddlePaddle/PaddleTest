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


class TestPrimitiveOp_6371bf470f19aad696f44517448194dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dfcedf316eacbbadf7dd6cacaded896
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.24757005274295807]], [[0.15838868916034698]], [[0.20460423827171326]], [[0.22411410510540009]], [[0.05801490694284439]], [[0.20261166989803314]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.70428466796875]], [[0.5585848689079285]], [[0.5928927659988403]], [[0.7616262435913086]], [[0.6079586148262024]], [[0.515092670917511]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_529d9c2554e8a0ac76048a4878f2d391(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dfcedf316eacbbadf7dd6cacaded896
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.011824175715446472]], [[0.45270979404449463]], [[0.34316617250442505]], [[0.2273150533437729]], [[0.18783067166805267]], [[0.3581860065460205]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6252842545509338]], [[0.7332195043563843]], [[0.7346566915512085]], [[0.7151126861572266]], [[0.8184952139854431]], [[0.7468409538269043]]], dtype='float32').reshape([6, 1, 1]),
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


class TestPrimitiveOp_563c01db0dc84003227f26876558aefa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2530a4e4625af688f4e3de9927836b64
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.30695095658302307, 0.45106175541877747]], [[0.20983338356018066, 0.39072245359420776]], [[0.01501217670738697, 0.2305171936750412]], [[0.498440682888031, 0.4070481061935425]], [[0.26861482858657837, 0.21705517172813416]], [[0.12172413617372513, 0.3923328220844269]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.12647491693496704, 0.4325127899646759]], [[0.023434879258275032, 0.31193414330482483]], [[0.29126760363578796, 0.25702059268951416]], [[0.22428520023822784, 0.029101457446813583]], [[0.4174845218658447, 0.44146567583084106]], [[0.08141609281301498, 0.07235391438007355]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_5f05c884f1414a76ef047ac96334222c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2530a4e4625af688f4e3de9927836b64
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.28746286034584045, 0.3657473921775818]], [[0.06539379805326462, 0.02533445507287979]], [[0.45328935980796814, 0.4643558859825134]], [[0.11082705110311508, 0.4665272533893585]], [[0.39100441336631775, 0.1641365885734558]], [[0.34363889694213867, 0.3672163784503937]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.12647491693496704, 0.4325127899646759]], [[0.023434879258275032, 0.31193414330482483]], [[0.29126760363578796, 0.25702059268951416]], [[0.22428520023822784, 0.029101457446813583]], [[0.4174845218658447, 0.44146567583084106]], [[0.08141609281301498, 0.07235391438007355]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class TestPrimitiveOp_f25edcd9533fb9c5b982040fa43708c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38e60ad0c3265f689e713b6b70f444bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2598961591720581, 0.43269333243370056]], [[0.4344225227832794, 0.22085875272750854]], [[0.3970338702201843, 0.2212526798248291]], [[0.11331501603126526, 0.4938947856426239]], [[0.34361928701400757, 0.33931711316108704]], [[0.19272422790527344, 0.41863080859184265]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class TestPrimitiveOp_1539b6fa38461fe6532325e8241d1819(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0368fad43623b73e790d949bddeb3e4a
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
            paddle.to_tensor([0.1883522868156433, 0.4092213809490204, 0.1282816231250763, 0.42731329798698425, 0.26688024401664734, 0.4253476560115814, 0.47656574845314026, 0.4448254108428955, 0.023841800168156624, 0.15511144697666168, 0.3392214775085449, 0.4359172284603119, 0.4742584228515625, 0.024728234857320786, 0.40860405564308167, 0.3769502341747284], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_248c483d917084c0e34f8e2a2a614da2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0368fad43623b73e790d949bddeb3e4a
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1883522868156433, 0.4092213809490204, 0.1282816231250763, 0.42731329798698425, 0.26688024401664734, 0.4253476560115814, 0.47656574845314026, 0.4448254108428955, 0.023841800168156624, 0.15511144697666168, 0.3392214775085449, 0.4359172284603119, 0.4742584228515625, 0.024728234857320786, 0.40860405564308167, 0.3769502341747284], dtype='float32').reshape([16]),
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


class PrimitiveOp_c38f93566d73f273afb1185373f5037c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1786, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1786, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5afe55c50151db281933ff551f248ae7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c38f93566d73f273afb1185373f5037c
    def get_inputs(self):
        return [
            paddle.uniform([1786, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2119479ec54290bfbe0f5a69e8f3c3d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1786, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1786, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_10bc4c0addb13c72aec40c36b71c7c1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2119479ec54290bfbe0f5a69e8f3c3d7
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10bc4c0addb13c72aec40c36b71c7c1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2119479ec54290bfbe0f5a69e8f3c3d7
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10bc4c0addb13c72aec40c36b71c7c1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2119479ec54290bfbe0f5a69e8f3c3d7
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10bc4c0addb13c72aec40c36b71c7c1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2119479ec54290bfbe0f5a69e8f3c3d7
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10bc4c0addb13c72aec40c36b71c7c1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2119479ec54290bfbe0f5a69e8f3c3d7
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10bc4c0addb13c72aec40c36b71c7c1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2119479ec54290bfbe0f5a69e8f3c3d7
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10bc4c0addb13c72aec40c36b71c7c1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2119479ec54290bfbe0f5a69e8f3c3d7
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10bc4c0addb13c72aec40c36b71c7c1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2119479ec54290bfbe0f5a69e8f3c3d7
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10bc4c0addb13c72aec40c36b71c7c1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2119479ec54290bfbe0f5a69e8f3c3d7
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10bc4c0addb13c72aec40c36b71c7c1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2119479ec54290bfbe0f5a69e8f3c3d7
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10bc4c0addb13c72aec40c36b71c7c1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2119479ec54290bfbe0f5a69e8f3c3d7
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_5afe55c50151db281933ff551f248ae7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c38f93566d73f273afb1185373f5037c
    def get_inputs(self):
        return [
            paddle.uniform([1786, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_28ba8fd94d10b25b59f4881b90fcdc3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_012768c0db613e5ecd733a6a0e0edd3f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.28419527411460876, 0.12107562273740768, 0.1396118551492691, 0.42913004755973816], [0.26608261466026306, 0.4100886285305023, 0.06463397294282913, 0.18227839469909668], [0.37692031264305115, 0.19566406309604645, 0.2698065936565399, 0.11681000143289566], [0.18090666830539703, 0.27563804388046265, 0.4358616769313812, 0.26883676648139954], [0.07208694517612457, 0.3047045171260834, 0.032146893441677094, 0.21066388487815857]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.05386225879192352, 0.26443421840667725, 0.3847455382347107, 0.4175401031970978], [0.08887602388858795, 0.0989922508597374, 0.35734865069389343, 0.17074978351593018], [0.12888197600841522, 0.08013017475605011, 0.32081493735313416, 0.4122312664985657], [0.15508875250816345, 0.2132154256105423, 0.11754553765058517, 0.15934787690639496], [0.19518782198429108, 0.467803955078125, 0.3062051236629486, 0.22477522492408752]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_312016bdf04504265d1f28ae231713c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_012768c0db613e5ecd733a6a0e0edd3f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.31687822937965393, 0.18115997314453125, 0.3580831289291382, 0.3002692461013794], [0.071229487657547, 0.24608850479125977, 0.4139639139175415, 0.42764025926589966], [0.4716216027736664, 0.3071999251842499, 0.0014963357243686914, 0.11614914983510971], [0.071229487657547, 0.24608850479125977, 0.4139639139175415, 0.42764025926589966], [0.4716216027736664, 0.3071999251842499, 0.0014963357243686914, 0.11614914983510971]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.4689908027648926, 0.31984302401542664, 0.280644953250885, 0.47177785634994507], [0.17404375970363617, 0.3382934331893921, 0.26541778445243835, 0.3190338909626007], [0.381242036819458, 0.3287785053253174, 0.0032447176054120064, 0.18423177301883698], [0.17404375970363617, 0.3382934331893921, 0.26541778445243835, 0.3190338909626007], [0.381242036819458, 0.3287785053253174, 0.0032447176054120064, 0.18423177301883698]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_8d201d10769fa948f2726d44ff1b819a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0944250226020813], [0.41575801372528076], [0.09537261724472046], [0.13353395462036133], [0.017560848966240883], [0.10870978981256485], [0.15444859862327576], [0.14588458836078644], [0.06295822560787201]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.2622634470462799], [0.2463095784187317], [0.40173643827438354], [0.3592231869697571], [0.3121943771839142], [0.22684775292873383], [0.2601875066757202], [0.3055746257305145], [0.2560676038265228]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_e22b5736e748215131140a9426801226(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15379993617534637], [0.05409188196063042], [0.31611233949661255], [0.3350580632686615], [0.24807313084602356], [0.003155889455229044], [0.12327843904495239], [0.15179887413978577], [0.044809091836214066]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3488651216030121], [0.4772522747516632], [0.08202075958251953], [0.36751043796539307], [0.43682003021240234], [0.356018990278244], [0.25737234950065613], [0.4559156596660614], [0.34402239322662354]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_e44d74fc113581d025d85f440bed8f7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0944250226020813], [0.43199148774147034], [0.30663737654685974], [0.30789077281951904], [0.017560848966240883], [0.4452926218509674], [0.3800290524959564], [0.212552011013031], [0.06295822560787201]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.2622634470462799], [0.2463095784187317], [0.40173643827438354], [0.3592231869697571], [0.3121943771839142], [0.14940416812896729], [0.09890615940093994], [0.3055746257305145], [0.06899479031562805]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_5c54b51483195b0451f046d747f70f8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.46759042143821716], [0.05409188196063042], [0.31611233949661255], [0.4931334853172302], [0.3475216031074524], [0.33039727807044983], [0.12327843904495239], [0.307874858379364], [0.3336060643196106]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3488651216030121], [0.4772522747516632], [0.07919348776340485], [0.36751043796539307], [0.10987333208322525], [0.11728937923908234], [0.13010436296463013], [0.21030104160308838], [0.2969972491264343]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_8aaf9a332ee13b1615cb7d4a332e2ff1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4870230257511139], [0.41575801372528076], [0.09537261724472046], [0.13353395462036133], [0.23935119807720184], [0.10870978981256485], [0.15444859862327576], [0.14588458836078644], [0.23597005009651184]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.08312228322029114], [0.009725884534418583], [0.31125256419181824], [0.18358245491981506], [0.14136718213558197], [0.22684775292873383], [0.2601875066757202], [0.1880447119474411], [0.2560676038265228]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_b7c448506bfd361ff58b3fdcf1fdc7d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15379993617534637], [0.34884482622146606], [0.3303232789039612], [0.3350580632686615], [0.24807313084602356], [0.003155889455229044], [0.48330458998680115], [0.15179887413978577], [0.044809091836214066]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.07335024327039719], [0.072712242603302], [0.08202075958251953], [0.35215020179748535], [0.43682003021240234], [0.356018990278244], [0.25737234950065613], [0.4559156596660614], [0.34402239322662354]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_077d518d7069e0d1ecacd05223cca8f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.012567024677991867], [0.033545464277267456], [-0.07613429427146912], [-0.005593098234385252], [-0.0885133370757103], [0.10474269092082977], [-0.025808751583099365], [0.003745029680430889], [0.005792463663965464]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_5ada6bed89f792f0d757407884ec6799(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4870230257511139], [0.43199148774147034], [0.30663737654685974], [0.30789077281951904], [0.23935119807720184], [0.4452926218509674], [0.3800290524959564], [0.212552011013031], [0.23597005009651184]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.08312228322029114], [0.009725884534418583], [0.31125256419181824], [0.18358245491981506], [0.14136718213558197], [0.14940416812896729], [0.09890615940093994], [0.1880447119474411], [0.06899479031562805]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_e768d7ac226dd031b2471970458f6a38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.46759042143821716], [0.34884482622146606], [0.3303232789039612], [0.4931334853172302], [0.3475216031074524], [0.33039727807044983], [0.48330458998680115], [0.307874858379364], [0.3336060643196106]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.07335024327039719], [0.072712242603302], [0.07919348776340485], [0.35215020179748535], [0.10987333208322525], [0.11728937923908234], [0.13010436296463013], [0.21030104160308838], [0.2969972491264343]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_85b8d1e786e6ef168cd60ff2a4977a8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15923389792442322], [0.11660128831863403], [-0.0011590110370889306], [0.01752539537847042], [0.02328573353588581], [0.06305616348981857], [0.09929267317056656], [0.0023912705946713686], [0.0061127664521336555]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.012567024677991867], [0.033545464277267456], [-0.07613429427146912], [-0.005593098234385252], [-0.0885133370757103], [0.10474269092082977], [-0.025808751583099365], [0.003745029680430889], [0.005792463663965464]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_09e3838c23522ada6b5b0751c9530a8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bf1bf92fa97a5f107ae929d9fa75fb0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [-0.0], [-0.0], [-0.0], [0.0], [-0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.9210782051086426], [0.7123062014579773], [-64.68901824951172], [1.319142460823059], [4.801183223724365], [-0.6611015796661377], [1.2599259614944458], [-0.5661254525184631], [0.05239899083971977]], dtype='float32').reshape([9, 1]),
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


class TestPrimitiveOp_0ca734f0a19faef718ebc6b1b12ea6c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dfcedf316eacbbadf7dd6cacaded896
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.07375569641590118]], [[0.0879448652267456]], [[0.4162990152835846]], [[0.4001326858997345]], [[0.3266385495662689]], [[0.20461037755012512]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.7049703598022461]], [[0.8226004838943481]], [[0.5391098856925964]], [[0.5475325584411621]], [[0.5748777985572815]], [[0.8003376126289368]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_30ca3cdc9a2a4892b325543313358164(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dfcedf316eacbbadf7dd6cacaded896
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.23402853310108185]], [[0.24981211125850677]], [[0.1812887042760849]], [[0.2073003053665161]], [[0.33479824662208557]], [[0.44976329803466797]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.5529621243476868]], [[0.6618142127990723]], [[0.6048946976661682]], [[0.7230639457702637]], [[0.5504274368286133]], [[0.5951051115989685]]], dtype='float32').reshape([6, 1, 1]),
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


class PrimitiveOp_e163b1a08df17fb99fcc0fea53ef3be6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5529, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[5529, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_45291480328fc2b62a3380b4d3aa073d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e163b1a08df17fb99fcc0fea53ef3be6
    def get_inputs(self):
        return [
            paddle.uniform([5529, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2af7c40b7c2c57e2db3fe4a610d64a8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5529, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5529, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8a9a6c2ce437589ce8136d9eb788f183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af7c40b7c2c57e2db3fe4a610d64a8e
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a9a6c2ce437589ce8136d9eb788f183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af7c40b7c2c57e2db3fe4a610d64a8e
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a9a6c2ce437589ce8136d9eb788f183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af7c40b7c2c57e2db3fe4a610d64a8e
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a9a6c2ce437589ce8136d9eb788f183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af7c40b7c2c57e2db3fe4a610d64a8e
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a9a6c2ce437589ce8136d9eb788f183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af7c40b7c2c57e2db3fe4a610d64a8e
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a9a6c2ce437589ce8136d9eb788f183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af7c40b7c2c57e2db3fe4a610d64a8e
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a9a6c2ce437589ce8136d9eb788f183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af7c40b7c2c57e2db3fe4a610d64a8e
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a9a6c2ce437589ce8136d9eb788f183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af7c40b7c2c57e2db3fe4a610d64a8e
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a9a6c2ce437589ce8136d9eb788f183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af7c40b7c2c57e2db3fe4a610d64a8e
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a9a6c2ce437589ce8136d9eb788f183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af7c40b7c2c57e2db3fe4a610d64a8e
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a9a6c2ce437589ce8136d9eb788f183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af7c40b7c2c57e2db3fe4a610d64a8e
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_45291480328fc2b62a3380b4d3aa073d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e163b1a08df17fb99fcc0fea53ef3be6
    def get_inputs(self):
        return [
            paddle.uniform([5529, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f0c05d843c5bee2599c90ee469e6a7cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2f2b4663ad0e38d2e5e931474ae84ac
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.23478740453720093, 0.3324480950832367, 0.3258914649486542, 0.21546480059623718], [0.03194620460271835, 0.3883354663848877, 0.41174057126045227, 0.16557937860488892], [0.32679083943367004, 0.0702432170510292, 0.4551614224910736, 0.06928850710391998], [0.03194620460271835, 0.3883354663848877, 0.41174057126045227, 0.16557937860488892], [0.32679083943367004, 0.0702432170510292, 0.4551614224910736, 0.06928850710391998], [0.21980880200862885, 0.4097957909107208, 0.38588380813598633, 0.11235152184963226], [0.21980880200862885, 0.4097957909107208, 0.38588380813598633, 0.11235152184963226]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.414839506149292, 0.3466450870037079, 0.16955099999904633, 0.11322132498025894], [0.3215886056423187, 0.36982491612434387, 0.37840282917022705, 0.0600690133869648], [0.043196480721235275, 0.16227102279663086, 0.18102847039699554, 0.14546465873718262], [0.3215886056423187, 0.36982491612434387, 0.37840282917022705, 0.0600690133869648], [0.043196480721235275, 0.16227102279663086, 0.18102847039699554, 0.14546465873718262], [0.4184300899505615, 0.18793024122714996, 0.17538608610630035, 0.018033312633633614], [0.4184300899505615, 0.18793024122714996, 0.17538608610630035, 0.018033312633633614]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_fc7d0f9827dc9aa2d187e8a91a9ff18c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4773276150226593, 0.4900837242603302, 0.31976962089538574, 0.23725539445877075, 0.3398610055446625, 0.47337961196899414], dtype='float32').reshape([6]),
            paddle.to_tensor([0.09849236905574799, 0.377002090215683, 0.03728121519088745, 0.03544168174266815, 0.3461141586303711, 0.37609827518463135], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_45ddbf1aa089b68f7c8cec16f99ba70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12860596179962158, 0.07729382812976837, 0.4407457411289215, 0.08533225953578949, 0.47597774863243103, 0.44470906257629395], dtype='float32').reshape([6]),
            paddle.to_tensor([0.10391133278608322, 0.45978280901908875, 0.49666598439216614, 0.21115154027938843, 0.31940513849258423, 0.33928123116493225], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_fe3f96ae6b9dbbb68edbffbc9b0570ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09207890182733536, 0.23877081274986267, 0.14726030826568604, 0.3021329939365387, 0.4564148485660553, 0.3768220543861389], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2667119801044464, 0.46009713411331177, 0.48630520701408386, 0.3197534382343292, 0.08830360323190689, 0.18910035490989685], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_d1f8f86db0b67dd20de38989ce0b1f63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.32528331875801086, 0.010651743039488792, 0.30837002396583557, 0.4661112129688263, 0.16577965021133423, 0.012630387209355831], dtype='float32').reshape([6]),
            paddle.to_tensor([0.33211463689804077, 0.06320647895336151, 0.03932555019855499, 0.3673308491706848, 0.05595792829990387, 0.18198029696941376], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_a4401bc6c30780b17391b247a7bbd38c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09207890182733536, 0.23877081274986267, 0.14726030826568604, 0.23725539445877075, 0.3461141586303711, 0.3768220543861389], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2667119801044464, 0.46009713411331177, 0.48630520701408386, 0.3197534382343292, 0.3461141586303711, 0.37609827518463135], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_d72adaa9d9b47f97e19139c777cb3a9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12860596179962158, 0.010651743039488792, 0.30837002396583557, 0.21115154027938843, 0.16577965021133423, 0.012630387209355831], dtype='float32').reshape([6]),
            paddle.to_tensor([0.33211463689804077, 0.45978280901908875, 0.49666598439216614, 0.3673308491706848, 0.31940513849258423, 0.33928123116493225], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_3b4ee69f2a16430aafe15af2084d9a2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4773276150226593, 0.4900837242603302, 0.31976962089538574, 0.23725539445877075, 0.3461141586303711, 0.47337961196899414], dtype='float32').reshape([6]),
            paddle.to_tensor([0.09849236905574799, 0.377002090215683, 0.03728121519088745, 0.03544168174266815, 0.3461141586303711, 0.37609827518463135], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_b4e71e63b1fde41d1b8e9c3bd407af4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12860596179962158, 0.45978280901908875, 0.49666598439216614, 0.21115154027938843, 0.47597774863243103, 0.44470906257629395], dtype='float32').reshape([6]),
            paddle.to_tensor([0.10391133278608322, 0.45978280901908875, 0.49666598439216614, 0.21115154027938843, 0.31940513849258423, 0.33928123116493225], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ea1d9f56fafb3930e90a81a8ad857485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.01054816972464323, 0.011631745845079422, -0.09121815115213394, -0.0017405538819730282, 0.04042661190032959, -0.021534491330385208], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, -0.0, -0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_4bc3fd73ea19907ac90994cd3a59cfaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.28790998458862305, 0.4335429072380066, 0.1785254180431366, 0.13634854555130005, 0.342987596988678, 0.42473894357681274], dtype='float32').reshape([6]),
            paddle.to_tensor([0.17939543724060059, 0.349433958530426, 0.31678277254104614, 0.31094321608543396, 0.2723592221736908, 0.2829611897468567], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_14e13814c62f5d82d731120f7cb50d70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1162586510181427, 0.26853832602500916, 0.4687058627605438, 0.14824190735816956, 0.39769142866134644, 0.3919951319694519], dtype='float32').reshape([6]),
            paddle.to_tensor([0.328698992729187, 0.036929111927747726, 0.17384779453277588, 0.41672104597091675, 0.11086878925561905, 0.09730534255504608], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_2bd7274cd8aecac9a9b39ccd1b31daf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4773276150226593, 0.4900837242603302, 0.31976962089538574, 0.3021329939365387, 0.4564148485660553, 0.47337961196899414], dtype='float32').reshape([6]),
            paddle.to_tensor([0.09849236905574799, 0.377002090215683, 0.03728121519088745, 0.03544168174266815, 0.08830360323190689, 0.18910035490989685], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_a211eecaeeea7b04fb83d20e17e4da6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([0.32528331875801086, 0.45978280901908875, 0.49666598439216614, 0.4661112129688263, 0.47597774863243103, 0.44470906257629395], dtype='float32').reshape([6]),
            paddle.to_tensor([0.10391133278608322, 0.06320647895336151, 0.03932555019855499, 0.21115154027938843, 0.05595792829990387, 0.18198029696941376], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_60698c18d247628df2d3c0de817aeb40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62f3ea2318dbf64f3e7d739d508b025
    def get_inputs(self):
        return [
            paddle.to_tensor([1.531698226928711, 1.3376604318618774, -0.9000090956687927, -0.176523357629776, 1.2808647155761719, -0.8368041515350342], dtype='float32').reshape([6]),
            paddle.to_tensor([1.5057028532028198, -0.28745824098587036, -1.3753670454025269, -1.0133177042007446, -0.0399165078997612, 0.7452316880226135], dtype='float32').reshape([6]),
        ]


class PrimitiveOp_934b66a6f6c89d30c3aa9cdc72474975(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1767, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1767, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_886f9de4be28bc651cf86e6e8383595c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_934b66a6f6c89d30c3aa9cdc72474975
    def get_inputs(self):
        return [
            paddle.uniform([1767, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c64d479badb8c81146db5594601aeab0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1767, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1767, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1123cc5835b4c1b6fc402803a10b8dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c64d479badb8c81146db5594601aeab0
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1123cc5835b4c1b6fc402803a10b8dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c64d479badb8c81146db5594601aeab0
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1123cc5835b4c1b6fc402803a10b8dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c64d479badb8c81146db5594601aeab0
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1123cc5835b4c1b6fc402803a10b8dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c64d479badb8c81146db5594601aeab0
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1123cc5835b4c1b6fc402803a10b8dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c64d479badb8c81146db5594601aeab0
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1123cc5835b4c1b6fc402803a10b8dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c64d479badb8c81146db5594601aeab0
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1123cc5835b4c1b6fc402803a10b8dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c64d479badb8c81146db5594601aeab0
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1123cc5835b4c1b6fc402803a10b8dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c64d479badb8c81146db5594601aeab0
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1123cc5835b4c1b6fc402803a10b8dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c64d479badb8c81146db5594601aeab0
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1123cc5835b4c1b6fc402803a10b8dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c64d479badb8c81146db5594601aeab0
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1123cc5835b4c1b6fc402803a10b8dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c64d479badb8c81146db5594601aeab0
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_886f9de4be28bc651cf86e6e8383595c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_934b66a6f6c89d30c3aa9cdc72474975
    def get_inputs(self):
        return [
            paddle.uniform([1767, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_bce3d6aec9d082565ee597ee79e5d2e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0df867a4644c834ec9fd270790bdf2c
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4294268786907196, 0.22505371272563934, 0.11226247251033783, 0.40070340037345886, 0.2896074652671814, 0.39202064275741577, 0.26139435172080994, 0.4006882309913635, 0.23677672445774078, 0.27721020579338074, 0.3967859148979187, 0.12629766762256622, 0.062405284494161606, 0.47626909613609314, 0.055194057524204254, 0.27083051204681396, 0.395824670791626, 0.06234115734696388, 0.14715760946273804, 0.2561332881450653, 0.4737702012062073, 0.19003598392009735, 0.03270141780376434, 0.3108614385128021], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_84313a0740448595e42c21c10d5af915(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0df867a4644c834ec9fd270790bdf2c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4294268786907196, 0.22505371272563934, 0.11226247251033783, 0.40070340037345886, 0.2896074652671814, 0.39202064275741577, 0.26139435172080994, 0.4006882309913635, 0.23677672445774078, 0.27721020579338074, 0.3967859148979187, 0.12629766762256622, 0.062405284494161606, 0.47626909613609314, 0.055194057524204254, 0.27083051204681396, 0.395824670791626, 0.06234115734696388, 0.14715760946273804, 0.2561332881450653, 0.4737702012062073, 0.19003598392009735, 0.03270141780376434, 0.3108614385128021], dtype='float32').reshape([24]),
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


class TestPrimitiveOp_59582cb2e26093ad3c840c0b074b1bc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_473768ad5406d82abfbdd94c8502a0d7
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
            paddle.to_tensor([0.14466311037540436, 0.3338650166988373, 0.12534970045089722, 0.4669789671897888], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_82eac4e111542156639204fad3e6a5d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_473768ad5406d82abfbdd94c8502a0d7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14466311037540436, 0.3338650166988373, 0.12534970045089722, 0.4669789671897888], dtype='float32').reshape([4]),
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


class TestPrimitiveOp_39b1db1b775ba8eac9562035ac86bb17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93bf25610fd14d81a6b319a42245208a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.25113505125045776, 0.17078417539596558, 0.09534557908773422, 0.4018389880657196], [0.22085505723953247, 0.38356485962867737, 0.18432451784610748, 0.20863646268844604], [0.4819970428943634, 0.05819538235664368, 0.29151207208633423, 0.09675493836402893], [0.06634899228811264, 0.274440735578537, 0.30541256070137024, 0.26725640892982483], [0.06634899228811264, 0.274440735578537, 0.30541256070137024, 0.26725640892982483], [0.4819970428943634, 0.05819538235664368, 0.29151207208633423, 0.09675493836402893]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.13906514644622803, 0.37058523297309875, 0.3800674080848694, 0.49303969740867615], [0.2805522084236145, 0.23713815212249756, 0.0843280702829361, 0.4226423501968384], [0.30662262439727783, 0.4646814465522766, 0.21752184629440308, 0.01768474467098713], [0.198168084025383, 0.13198967278003693, 0.2838340699672699, 0.11747059971094131], [0.198168084025383, 0.13198967278003693, 0.2838340699672699, 0.11747059971094131], [0.30662262439727783, 0.4646814465522766, 0.21752184629440308, 0.01768474467098713]], dtype='float32').reshape([6, 4]),
        ]


class TestPrimitiveOp_551f1619541343c2aaee538e89f99aef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_012768c0db613e5ecd733a6a0e0edd3f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12017253786325455, 0.3328137993812561, 0.2396862655878067, 0.0057601616717875], [0.09302746504545212, 0.3219456672668457, 0.13859473168849945, 0.11403941363096237], [0.48253023624420166, 0.07991086691617966, 0.3013501763343811, 0.1660875827074051], [0.06722715497016907, 0.22898554801940918, 0.3784162700176239, 0.08708702772855759], [0.12017253786325455, 0.3328137993812561, 0.2396862655878067, 0.0057601616717875]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.10527151077985764, 0.327819287776947, 0.23980413377285004, 0.18861910700798035], [0.2173198163509369, 0.2349286824464798, 0.16776219010353088, 0.13265608251094818], [0.37160301208496094, 0.18354126811027527, 0.4221250116825104, 0.2687823176383972], [0.49743491411209106, 0.3538717031478882, 0.43951812386512756, 0.07650578022003174], [0.10527151077985764, 0.327819287776947, 0.23980413377285004, 0.18861910700798035]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_e65c1085c818e27b093100bd1cd0c39f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3083866834640503]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3151826858520508]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_6127b9e9e20b0ccdf22a0be20c897309(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16263581812381744]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.21531380712985992]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_92b994c7e2ea9b76b31eed2fae3bf8df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.41494059562683105]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.10528379678726196]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_e85216b012d92ee4e6f6270204c0cc6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16263581812381744]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.07341162115335464]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_e65c1085c818e27b093100bd1cd0c39f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3083866834640503]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3151826858520508]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_7a89baf4c3505b42c4be5b7c2e0d0926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3545854091644287]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.21531380712985992]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_dfd2f34f19b098e954222d079dc58924(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.02668238990008831]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_92b994c7e2ea9b76b31eed2fae3bf8df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.41494059562683105]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.10528379678726196]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_8f8ad69d94a49352fe21f874232bbbed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3545854091644287]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.07341162115335464]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_57632534d760a09165510a6836260d33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0870673805475235]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.02668238990008831]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_23d08716d57d9f7b9ca2afe985875ff1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a70e9966e0478e7bd89c97e9806e0ca
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.6935431957244873]], dtype='float32').reshape([1, 1]),
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


class TestPrimitiveOp_31e27abf55bc03c2d67f1b19e6d4accf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.04222099483013153], [0.45837101340293884], [0.4508039653301239], [0.24826665222644806], [0.04853551834821701], [0.25887274742126465]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.10129960626363754], [0.4956405460834503], [0.3438938558101654], [0.49526292085647583], [0.2497348189353943], [0.3659272789955139]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_2a57e9831b660c3bf8a1b8fcb63f4496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2849709689617157], [0.11111163347959518], [0.23699478805065155], [0.22173336148262024], [0.05751029774546623], [0.04977961629629135]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3794229328632355], [0.35555100440979004], [0.2228093296289444], [0.22192063927650452], [0.43225693702697754], [0.41214850544929504]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_47b3c01f8bc9f6aad4fd46f0556f8634(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.376817524433136], [0.45837101340293884], [0.4508039653301239], [0.24826665222644806], [0.04853551834821701], [0.25887274742126465]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.10129960626363754], [0.42914801836013794], [0.3438938558101654], [0.010731011629104614], [0.2497348189353943], [0.2733650505542755]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_0a2269454309ede07b0d5c50dd84bcf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2849709689617157], [0.4665500521659851], [0.4074956178665161], [0.22639164328575134], [0.21449042856693268], [0.04977961629629135]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.31633827090263367], [0.34771016240119934], [0.2228093296289444], [0.22192063927650452], [0.43225693702697754], [0.3838341534137726]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_0790e09355918d8b156c7d6fb0400555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.04222099483013153], [0.4947202205657959], [0.46678364276885986], [0.2570192813873291], [0.16303293406963348], [0.3793601393699646]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.09891922771930695], [0.4956405460834503], [0.13846367597579956], [0.49526292085647583], [0.09620732814073563], [0.3659272789955139]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_2e385d0fa1565c706fcaf2c7c053002e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3085649907588959], [0.11111163347959518], [0.23699478805065155], [0.22173336148262024], [0.05751029774546623], [0.39112553000450134]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3794229328632355], [0.35555100440979004], [0.0830652266740799], [0.07343913614749908], [0.06483077257871628], [0.41214850544929504]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_f388a593bdd49a50d38f4334ab31f392(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0046247332356870174], [0.003697821171954274], [0.07028298079967499], [-0.03426813334226608], [0.043325275182724], [0.004558820743113756]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0015165689401328564], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_e90de5e11718671e7e00799c23424e71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.376817524433136], [0.4947202205657959], [0.46678364276885986], [0.2570192813873291], [0.16303293406963348], [0.3793601393699646]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.09891922771930695], [0.42914801836013794], [0.13846367597579956], [0.010731011629104614], [0.09620732814073563], [0.2733650505542755]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_169cc61d0a10b9cd119b77bf084abd32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3085649907588959], [0.4665500521659851], [0.4074956178665161], [0.22639164328575134], [0.21449042856693268], [0.39112553000450134]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.31633827090263367], [0.34771016240119934], [0.0830652266740799], [0.07343913614749908], [0.06483077257871628], [0.3838341534137726]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_201cc864f67ade74d5863cddf7379aed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.00216018152423203], [0.0077925934456288815], [0.10651697963476181], [0.03767040744423866], [0.010001097805798054], [0.0007728502387180924]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.0046247332356870174], [0.003697821171954274], [0.06876641511917114], [-0.03426813334226608], [0.043325275182724], [0.004558820743113756]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_242f333c5472ed2ab2aca89ff70f54bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8293d79d107f0daf7a7b6dff16e1dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [0.0], [0.022053919732570648], [-0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-1.1409002542495728], [0.5254697203636169], [0.3544088900089264], [1.9096832275390625], [-3.332051992416382], [-4.898711681365967]], dtype='float32').reshape([6, 1]),
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


class TestPrimitiveOp_72b5969c185fa983084450f6cbcd42d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e223e011a18f61a428db8a806582d14
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08920494467020035, 0.46364647150039673, 0.03781019523739815, 0.4345451593399048], [0.47518661618232727, 0.3344584107398987, 0.490209698677063, 0.4670402705669403], [0.22168733179569244, 0.3785659670829773, 0.1331939548254013, 0.37388402223587036], [0.4551789462566376, 0.2746458053588867, 0.4354214072227478, 0.4967152178287506]], dtype='float32').reshape([4, 4]),
            paddle.to_tensor([[0.48525816202163696, 0.1439717561006546, 0.06915892660617828, 0.28783324360847473], [0.038753412663936615, 0.3068057596683502, 0.3743298351764679, 0.3719959557056427], [0.28154608607292175, 0.030832892283797264, 0.3473222851753235, 0.0951429158449173], [0.1881830245256424, 0.18366138637065887, 0.42459896206855774, 0.4346340000629425]], dtype='float32').reshape([4, 4]),
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


class PrimitiveOp_9bcd4967652e831823a7760c11739649(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2010, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2010, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b1c48f18a1a7947f7002d737594842e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bcd4967652e831823a7760c11739649
    def get_inputs(self):
        return [
            paddle.uniform([2010, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_041df54bd1da3cbc8e51ff4472da2890(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2010, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2010, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d8b4639b0bc9f163676ecd4a6e257e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_041df54bd1da3cbc8e51ff4472da2890
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d8b4639b0bc9f163676ecd4a6e257e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_041df54bd1da3cbc8e51ff4472da2890
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d8b4639b0bc9f163676ecd4a6e257e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_041df54bd1da3cbc8e51ff4472da2890
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d8b4639b0bc9f163676ecd4a6e257e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_041df54bd1da3cbc8e51ff4472da2890
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d8b4639b0bc9f163676ecd4a6e257e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_041df54bd1da3cbc8e51ff4472da2890
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d8b4639b0bc9f163676ecd4a6e257e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_041df54bd1da3cbc8e51ff4472da2890
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d8b4639b0bc9f163676ecd4a6e257e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_041df54bd1da3cbc8e51ff4472da2890
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d8b4639b0bc9f163676ecd4a6e257e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_041df54bd1da3cbc8e51ff4472da2890
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d8b4639b0bc9f163676ecd4a6e257e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_041df54bd1da3cbc8e51ff4472da2890
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d8b4639b0bc9f163676ecd4a6e257e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_041df54bd1da3cbc8e51ff4472da2890
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d8b4639b0bc9f163676ecd4a6e257e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_041df54bd1da3cbc8e51ff4472da2890
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_b1c48f18a1a7947f7002d737594842e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bcd4967652e831823a7760c11739649
    def get_inputs(self):
        return [
            paddle.uniform([2010, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_703e7cba8e816634c4751e27aa2ecddf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2f2b4663ad0e38d2e5e931474ae84ac
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.28683289885520935, 0.24949973821640015, 0.29143574833869934, 0.07112333923578262], [0.28683289885520935, 0.24949973821640015, 0.29143574833869934, 0.07112333923578262], [0.3108574151992798, 0.0966305360198021, 0.1343512088060379, 0.2716635465621948], [0.20311346650123596, 0.3568199574947357, 0.19041480123996735, 0.41252681612968445], [0.3272130489349365, 0.2632125914096832, 0.44407472014427185, 0.05285264179110527], [0.02742091752588749, 0.024939358234405518, 0.04848539084196091, 0.19067732989788055], [0.20881330966949463, 0.22709128260612488, 0.2838110625743866, 0.19150592386722565]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.357042521238327, 0.01976151578128338, 0.49920791387557983, 0.3945619761943817], [0.357042521238327, 0.01976151578128338, 0.49920791387557983, 0.3945619761943817], [0.44208860397338867, 0.3043256998062134, 0.47371426224708557, 0.07150912284851074], [0.4755665063858032, 0.3525017499923706, 0.12312207370996475, 0.016796007752418518], [0.17840193212032318, 0.47789740562438965, 0.39662763476371765, 0.11381888389587402], [0.056634217500686646, 0.48900488018989563, 0.30459845066070557, 0.48191460967063904], [0.20183145999908447, 0.16153369843959808, 0.19585464894771576, 0.28988736867904663]], dtype='float32').reshape([7, 4]),
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


class PrimitiveOp_e2d555acbde5bc9ef5f256bf1b70d365(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4663, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[4663, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_352387062433ce0edf5c2a45c90a472b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2d555acbde5bc9ef5f256bf1b70d365
    def get_inputs(self):
        return [
            paddle.uniform([4663, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e3b172a9590dc3b5aebfea802723c55b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4663, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4663, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b21239a8d541819f68cdbfdcb675ccf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3b172a9590dc3b5aebfea802723c55b
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b21239a8d541819f68cdbfdcb675ccf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3b172a9590dc3b5aebfea802723c55b
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b21239a8d541819f68cdbfdcb675ccf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3b172a9590dc3b5aebfea802723c55b
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b21239a8d541819f68cdbfdcb675ccf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3b172a9590dc3b5aebfea802723c55b
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b21239a8d541819f68cdbfdcb675ccf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3b172a9590dc3b5aebfea802723c55b
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b21239a8d541819f68cdbfdcb675ccf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3b172a9590dc3b5aebfea802723c55b
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b21239a8d541819f68cdbfdcb675ccf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3b172a9590dc3b5aebfea802723c55b
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b21239a8d541819f68cdbfdcb675ccf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3b172a9590dc3b5aebfea802723c55b
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b21239a8d541819f68cdbfdcb675ccf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3b172a9590dc3b5aebfea802723c55b
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b21239a8d541819f68cdbfdcb675ccf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3b172a9590dc3b5aebfea802723c55b
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b21239a8d541819f68cdbfdcb675ccf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3b172a9590dc3b5aebfea802723c55b
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_352387062433ce0edf5c2a45c90a472b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2d555acbde5bc9ef5f256bf1b70d365
    def get_inputs(self):
        return [
            paddle.uniform([4663, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_261107d0d7709920067ea3e1b1c1884c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1090, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1090, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_467a7f5475774e83ff3526172b94fcd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_261107d0d7709920067ea3e1b1c1884c
    def get_inputs(self):
        return [
            paddle.uniform([1090, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c365186bb23637b71f6d66a41b0450a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1090, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1090, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1663f6a78f4432e6f0a60570f74eca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c365186bb23637b71f6d66a41b0450a8
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1663f6a78f4432e6f0a60570f74eca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c365186bb23637b71f6d66a41b0450a8
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1663f6a78f4432e6f0a60570f74eca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c365186bb23637b71f6d66a41b0450a8
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1663f6a78f4432e6f0a60570f74eca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c365186bb23637b71f6d66a41b0450a8
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1663f6a78f4432e6f0a60570f74eca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c365186bb23637b71f6d66a41b0450a8
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1663f6a78f4432e6f0a60570f74eca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c365186bb23637b71f6d66a41b0450a8
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1663f6a78f4432e6f0a60570f74eca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c365186bb23637b71f6d66a41b0450a8
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1663f6a78f4432e6f0a60570f74eca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c365186bb23637b71f6d66a41b0450a8
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1663f6a78f4432e6f0a60570f74eca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c365186bb23637b71f6d66a41b0450a8
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1663f6a78f4432e6f0a60570f74eca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c365186bb23637b71f6d66a41b0450a8
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1663f6a78f4432e6f0a60570f74eca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c365186bb23637b71f6d66a41b0450a8
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_467a7f5475774e83ff3526172b94fcd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_261107d0d7709920067ea3e1b1c1884c
    def get_inputs(self):
        return [
            paddle.uniform([1090, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_97e9eb3da3d634cbe60a524bf7a8a50f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93bf25610fd14d81a6b319a42245208a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.27982795238494873, 0.23049882054328918, 0.12003326416015625, 0.31042230129241943], [0.27934736013412476, 0.2104044407606125, 0.14274001121520996, 0.43509340286254883], [0.27934736013412476, 0.2104044407606125, 0.14274001121520996, 0.43509340286254883], [0.06633070856332779, 0.42473211884498596, 0.07029067724943161, 0.18816295266151428], [0.3314078152179718, 0.1529044508934021, 0.05097723379731178, 0.3056374788284302], [0.026664773002266884, 0.43551871180534363, 0.3431084156036377, 0.43823882937431335]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.10475707799196243, 0.006118948571383953, 0.36331644654273987, 0.02887890860438347], [0.3239193260669708, 0.39356729388237, 0.24005842208862305, 0.09929648786783218], [0.3239193260669708, 0.39356729388237, 0.24005842208862305, 0.09929648786783218], [0.1777137666940689, 0.09029393643140793, 0.07026015967130661, 0.035986095666885376], [0.2511315643787384, 0.4074481725692749, 0.1954677850008011, 0.0010132929310202599], [0.18503394722938538, 0.207442045211792, 0.19336673617362976, 0.30191829800605774]], dtype='float32').reshape([6, 4]),
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


class TestPrimitiveOp_723d0879e77c2f49c44415832b5b2874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1546d5060eb5e7f91c606a3154f7d7bd
    def get_inputs(self):
        return [
            paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1.7267611026763916, 1.0060596466064453, 0.08557872474193573, 0.8079234957695007], [0.4078036844730377, 1.1999253034591675, 2.310272216796875, 0.9771278500556946]], dtype='float32').reshape([2, 4]),
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


class TestPrimitiveOp_7f880ef23f0bdc174f5bd6a475d6a9cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29eb514882c92ab3a86517239fc1edf9
    def get_inputs(self):
        return [
            paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1.655749797821045, 14.601186752319336, 0.35366272926330566, 1.535463571548462], [0.49857038259506226, 0.4970908761024475, 0.7121455073356628, 0.7435252070426941]], dtype='float32').reshape([2, 4]),
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


class TestPrimitiveOp_1a4406e6180f8e98b9d1fe603f8f98c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0807226225733757], [0.26894018054008484], [0.22837966680526733], [0.09048057347536087], [0.3214188814163208]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.35739219188690186], [0.11215049773454666], [0.29410940408706665], [0.08986014872789383], [0.42450934648513794]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_310d1cc0b58d5da23ab9bc059fe15c7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15623098611831665], [0.08885031193494797], [0.0010281642898917198], [0.315184623003006], [0.41620945930480957]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.34516990184783936], [0.4242517352104187], [0.3585534393787384], [0.4999895989894867], [0.33067619800567627]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_987f2115a7772fa25f0fc95366ceef00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0807226225733757], [0.26894018054008484], [0.30494368076324463], [0.27619537711143494], [0.428976833820343]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.06819140911102295], [0.0015212241560220718], [0.29410940408706665], [0.0312783420085907], [0.05078583583235741]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_e43643bc509d3676329fa2292fabdf46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.23073291778564453], [0.08885031193494797], [0.22880247235298157], [0.4462369680404663], [0.41620945930480957]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.34516990184783936], [0.14900963008403778], [0.23224982619285583], [0.4999895989894867], [0.029579993337392807]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_525aee59dfcdaec188beb9a308513a8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.42150676250457764], [0.40049660205841064], [0.22837966680526733], [0.09048057347536087], [0.3214188814163208]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.35739219188690186], [0.11215049773454666], [0.24850419163703918], [0.08986014872789383], [0.42450934648513794]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_ece8b3c4978108fc503147f09e5fba23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15623098611831665], [0.23012053966522217], [0.0010281642898917198], [0.315184623003006], [0.44070783257484436]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.2559264004230499], [0.4242517352104187], [0.3585534393787384], [0.4039344787597656], [0.33067619800567627]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_7eb4533e8a0937c836d63678512721cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.007825963199138641], [-0.07206472009420395], [0.007157676853239536], [-0.01321999728679657], [0.1348765641450882]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_2b1affb28e7b9af85e5a125e9ebed468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.42150676250457764], [0.40049660205841064], [0.30494368076324463], [0.27619537711143494], [0.428976833820343]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.06819140911102295], [0.0015212241560220718], [0.24850419163703918], [0.0312783420085907], [0.05078583583235741]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_99e85922a32431ddfac61fb43328ec25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.23073291778564453], [0.23012053966522217], [0.22880247235298157], [0.4462369680404663], [0.44070783257484436]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.2559264004230499], [0.14900963008403778], [0.23224982619285583], [0.4039344787597656], [0.029579993337392807]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_013632a85628cca45b3d0b162dcbc18d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.008901244029402733], [0.03236125409603119], [-0.00019456679001450539], [0.010360600426793098], [0.15548484027385712]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.007825963199138641], [-0.07206472009420395], [0.007157676853239536], [-0.01321999728679657], [0.1348765641450882]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_8ca970b1c92ab08c6da7ebb70f380a0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de5dca7af4528b013606c41f9180211
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0], [-0.0], [0.0], [-0.0], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.12080118805170059], [3.2268826961517334], [37.78776168823242], [2.2759876251220703], [0.13254202902317047]], dtype='float32').reshape([5, 1]),
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


class PrimitiveOp_c2fd63ea8f7dbf246d5cfc0de4c92581(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2374, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2374, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b0bf08a32823e8b4e0ac487409c2999b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fd63ea8f7dbf246d5cfc0de4c92581
    def get_inputs(self):
        return [
            paddle.uniform([2374, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_09f446c0811cc265129e8a175d97769d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2374, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2374, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_01bd7fb3c58f3e4450bfc64d70d7c3b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09f446c0811cc265129e8a175d97769d
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01bd7fb3c58f3e4450bfc64d70d7c3b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09f446c0811cc265129e8a175d97769d
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01bd7fb3c58f3e4450bfc64d70d7c3b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09f446c0811cc265129e8a175d97769d
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01bd7fb3c58f3e4450bfc64d70d7c3b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09f446c0811cc265129e8a175d97769d
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01bd7fb3c58f3e4450bfc64d70d7c3b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09f446c0811cc265129e8a175d97769d
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01bd7fb3c58f3e4450bfc64d70d7c3b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09f446c0811cc265129e8a175d97769d
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01bd7fb3c58f3e4450bfc64d70d7c3b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09f446c0811cc265129e8a175d97769d
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01bd7fb3c58f3e4450bfc64d70d7c3b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09f446c0811cc265129e8a175d97769d
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01bd7fb3c58f3e4450bfc64d70d7c3b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09f446c0811cc265129e8a175d97769d
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01bd7fb3c58f3e4450bfc64d70d7c3b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09f446c0811cc265129e8a175d97769d
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01bd7fb3c58f3e4450bfc64d70d7c3b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09f446c0811cc265129e8a175d97769d
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_b0bf08a32823e8b4e0ac487409c2999b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fd63ea8f7dbf246d5cfc0de4c92581
    def get_inputs(self):
        return [
            paddle.uniform([2374, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dcae0d5c3d9da70076c8d769b2721e34(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3058, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[3058, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_51475f4af1615a1289b62c0d24792e12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcae0d5c3d9da70076c8d769b2721e34
    def get_inputs(self):
        return [
            paddle.uniform([3058, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_373aef2afb7c95361696746d13854379(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3058, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3058, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_83cb9b96c3b6a0acd3b9a026f842a37f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_373aef2afb7c95361696746d13854379
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83cb9b96c3b6a0acd3b9a026f842a37f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_373aef2afb7c95361696746d13854379
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83cb9b96c3b6a0acd3b9a026f842a37f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_373aef2afb7c95361696746d13854379
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83cb9b96c3b6a0acd3b9a026f842a37f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_373aef2afb7c95361696746d13854379
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83cb9b96c3b6a0acd3b9a026f842a37f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_373aef2afb7c95361696746d13854379
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83cb9b96c3b6a0acd3b9a026f842a37f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_373aef2afb7c95361696746d13854379
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83cb9b96c3b6a0acd3b9a026f842a37f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_373aef2afb7c95361696746d13854379
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83cb9b96c3b6a0acd3b9a026f842a37f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_373aef2afb7c95361696746d13854379
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83cb9b96c3b6a0acd3b9a026f842a37f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_373aef2afb7c95361696746d13854379
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83cb9b96c3b6a0acd3b9a026f842a37f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_373aef2afb7c95361696746d13854379
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83cb9b96c3b6a0acd3b9a026f842a37f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_373aef2afb7c95361696746d13854379
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_51475f4af1615a1289b62c0d24792e12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcae0d5c3d9da70076c8d769b2721e34
    def get_inputs(self):
        return [
            paddle.uniform([3058, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fef8ed80e61b10b4b290fafe71fdead5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3793, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[3793, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b0e32ea5e696fe76b3ddb1f2b1618df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fef8ed80e61b10b4b290fafe71fdead5
    def get_inputs(self):
        return [
            paddle.uniform([3793, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a9c10a2aa97cfcaa3b2a9422932a351c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3793, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3793, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee07c4187ea4abd6318a9d83ce31ca1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c10a2aa97cfcaa3b2a9422932a351c
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee07c4187ea4abd6318a9d83ce31ca1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c10a2aa97cfcaa3b2a9422932a351c
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee07c4187ea4abd6318a9d83ce31ca1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c10a2aa97cfcaa3b2a9422932a351c
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee07c4187ea4abd6318a9d83ce31ca1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c10a2aa97cfcaa3b2a9422932a351c
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee07c4187ea4abd6318a9d83ce31ca1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c10a2aa97cfcaa3b2a9422932a351c
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee07c4187ea4abd6318a9d83ce31ca1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c10a2aa97cfcaa3b2a9422932a351c
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee07c4187ea4abd6318a9d83ce31ca1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c10a2aa97cfcaa3b2a9422932a351c
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee07c4187ea4abd6318a9d83ce31ca1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c10a2aa97cfcaa3b2a9422932a351c
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee07c4187ea4abd6318a9d83ce31ca1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c10a2aa97cfcaa3b2a9422932a351c
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee07c4187ea4abd6318a9d83ce31ca1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c10a2aa97cfcaa3b2a9422932a351c
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee07c4187ea4abd6318a9d83ce31ca1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9c10a2aa97cfcaa3b2a9422932a351c
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_8b0e32ea5e696fe76b3ddb1f2b1618df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fef8ed80e61b10b4b290fafe71fdead5
    def get_inputs(self):
        return [
            paddle.uniform([3793, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_fd56e4ee8207cc09ac1f007f597be06b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_042c31aed200f7f6f007e22c52c71138
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
            paddle.to_tensor([0.024768635630607605, 0.2426697313785553, 0.07255639135837555, 0.2981436252593994, 0.06335600465536118, 0.013720804825425148, 0.20028044283390045, 0.06637927889823914, 0.27754852175712585, 0.23517842590808868, 0.41420018672943115, 0.10023913532495499, 0.17942240834236145, 0.21710842847824097, 0.31704822182655334, 0.29076477885246277, 0.2488931566476822, 0.3848491609096527, 0.3888043463230133, 0.3472835123538971], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_244acf0142af88c9579bf7bc24e46a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_042c31aed200f7f6f007e22c52c71138
    def get_inputs(self):
        return [
            paddle.to_tensor([0.024768635630607605, 0.2426697313785553, 0.07255639135837555, 0.2981436252593994, 0.06335600465536118, 0.013720804825425148, 0.20028044283390045, 0.06637927889823914, 0.27754852175712585, 0.23517842590808868, 0.41420018672943115, 0.10023913532495499, 0.17942240834236145, 0.21710842847824097, 0.31704822182655334, 0.29076477885246277, 0.2488931566476822, 0.3848491609096527, 0.3888043463230133, 0.3472835123538971], dtype='float32').reshape([20]),
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


class TestPrimitiveOp_c1d2f2149bce067a7f7f3f4dc22553fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11390042304992676], [0.1851010024547577], [0.08248813450336456], [0.12678833305835724]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.339884489774704], [0.06699070334434509], [0.30670201778411865], [0.038516294211149216]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_2628da593704a02f1cf564f3f7d2b89b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.020291708409786224], [0.017377078533172607], [0.2168794572353363], [0.03630860522389412]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3211865723133087], [0.3201243281364441], [0.31398555636405945], [0.2734312415122986]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_e15dabaf6e1f9e41318b8a4c1f12d0d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11390042304992676], [0.4904465973377228], [0.13921834528446198], [0.496462345123291]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.06521501392126083], [0.034968309104442596], [0.26277071237564087], [0.01609121635556221]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_7c1647222084f793de74f7984e0d916a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3694194257259369], [0.4472231864929199], [0.41353461146354675], [0.07219047099351883]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.17842870950698853], [0.3201243281364441], [0.30915895104408264], [0.2734312415122986]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_7d47ee411ed287052030cfc942b911eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19393745064735413], [0.1851010024547577], [0.08248813450336456], [0.12678833305835724]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.339884489774704], [0.06699070334434509], [0.30670201778411865], [0.038516294211149216]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_160cdd0eeee97c08e153e8e36a820a04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.020291708409786224], [0.017377078533172607], [0.2168794572353363], [0.03630860522389412]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.3211865723133087], [0.02025376260280609], [0.31398555636405945], [0.023009276017546654]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_e0d740d70f57b6ecfe397cc5ce9b5619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.053213175386190414], [0.057551003992557526], [0.008876675739884377], [-0.0954962968826294]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_19c57ccb77a4be4b9a41a58a9efe0a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19393745064735413], [0.4904465973377228], [0.13921834528446198], [0.496462345123291]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.06521501392126083], [0.034968309104442596], [0.26277071237564087], [0.01609121635556221]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_29034d59eb65109aa1a94763aae2ff01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3694194257259369], [0.4472231864929199], [0.41353461146354675], [0.07219047099351883]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.17842870950698853], [0.02025376260280609], [0.30915895104408264], [0.023009276017546654]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_52edfd13242d9ec55a24f24e67538188(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.02458478882908821], [0.19447529315948486], [-0.012895859777927399], [0.023625224828720093]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.053213175386190414], [0.057551003992557526], [0.008876675739884377], [-0.0954962968826294]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_1d56ed12708e2384f53244233318c8ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c70bd2c1a88d3aacc2a656b1ddc09c
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [-0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-1.1644755601882935], [0.7040703892707825], [1.6883352994918823], [5.04213285446167]], dtype='float32').reshape([4, 1]),
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


class PrimitiveOp_ee2f950535f1e4a2db1535c66ba71c6c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2042, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2042, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60d2d3632fdcd031905fbd05ff65405d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee2f950535f1e4a2db1535c66ba71c6c
    def get_inputs(self):
        return [
            paddle.uniform([2042, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 4], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8f66e66798be4d62a2508c947bb5e00a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2042, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2042, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_28a12c13b855df56abd0092161ffefc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f66e66798be4d62a2508c947bb5e00a
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28a12c13b855df56abd0092161ffefc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f66e66798be4d62a2508c947bb5e00a
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28a12c13b855df56abd0092161ffefc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f66e66798be4d62a2508c947bb5e00a
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28a12c13b855df56abd0092161ffefc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f66e66798be4d62a2508c947bb5e00a
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28a12c13b855df56abd0092161ffefc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f66e66798be4d62a2508c947bb5e00a
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28a12c13b855df56abd0092161ffefc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f66e66798be4d62a2508c947bb5e00a
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28a12c13b855df56abd0092161ffefc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f66e66798be4d62a2508c947bb5e00a
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28a12c13b855df56abd0092161ffefc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f66e66798be4d62a2508c947bb5e00a
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28a12c13b855df56abd0092161ffefc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f66e66798be4d62a2508c947bb5e00a
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28a12c13b855df56abd0092161ffefc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f66e66798be4d62a2508c947bb5e00a
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28a12c13b855df56abd0092161ffefc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f66e66798be4d62a2508c947bb5e00a
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_60d2d3632fdcd031905fbd05ff65405d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee2f950535f1e4a2db1535c66ba71c6c
    def get_inputs(self):
        return [
            paddle.uniform([2042, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 4], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_ba9f6934073eb79adb8fb2b6ffa728eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_012768c0db613e5ecd733a6a0e0edd3f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.27431318163871765, 0.29873761534690857, 0.13117456436157227, 0.49010080099105835], [0.18418923020362854, 0.412865549325943, 0.17845621705055237, 0.08887195587158203], [0.387370228767395, 0.03513362631201744, 0.3818896412849426, 0.31303107738494873], [0.387370228767395, 0.03513362631201744, 0.3818896412849426, 0.31303107738494873], [0.23536142706871033, 0.07056697458028793, 0.06838814169168472, 0.07380599528551102]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.366535484790802, 0.032734017819166183, 0.36983925104141235, 0.0076654027216136456], [0.41283977031707764, 0.4335724413394928, 0.08661874383687973, 0.1704954355955124], [0.012915438041090965, 0.15451563894748688, 0.2023535966873169, 0.16199228167533875], [0.012915438041090965, 0.15451563894748688, 0.2023535966873169, 0.16199228167533875], [0.25577086210250854, 0.1705213487148285, 0.38791191577911377, 0.48989036679267883]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_dd41ba94a8cdbf854db40dd23998baa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2f2b4663ad0e38d2e5e931474ae84ac
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4940308630466461, 0.07509899139404297, 0.38665270805358887, 0.1330437958240509], [0.12829552590847015, 0.25552383065223694, 0.27220097184181213, 0.28099772334098816], [0.20824433863162994, 0.399840772151947, 0.2727225422859192, 0.14007455110549927], [0.4940308630466461, 0.07509899139404297, 0.38665270805358887, 0.1330437958240509], [0.08490654826164246, 0.08925537765026093, 0.27890777587890625, 0.19405165314674377], [0.42623135447502136, 0.29268595576286316, 0.2989133894443512, 0.10139984637498856], [0.08490654826164246, 0.08925537765026093, 0.27890777587890625, 0.19405165314674377]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.237456813454628, 0.36624595522880554, 0.11861702799797058, 0.30560624599456787], [0.4004635512828827, 0.03755925968289375, 0.03999197855591774, 0.46264731884002686], [0.14004486799240112, 0.415873646736145, 0.3796509802341461, 0.057666048407554626], [0.237456813454628, 0.36624595522880554, 0.11861702799797058, 0.30560624599456787], [0.2706938683986664, 0.32761648297309875, 0.2390376478433609, 0.04281473904848099], [0.08460255712270737, 0.18095530569553375, 0.09658311307430267, 0.19950956106185913], [0.2706938683986664, 0.32761648297309875, 0.2390376478433609, 0.04281473904848099]], dtype='float32').reshape([7, 4]),
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