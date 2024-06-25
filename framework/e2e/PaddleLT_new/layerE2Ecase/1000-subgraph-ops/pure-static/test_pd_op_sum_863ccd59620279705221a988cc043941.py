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



class PrimitiveOp_20abee4dad939004c06936a96334fcf3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 16, 9, 112, 112], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bdaaa2debef0ac06f30d42335a97cc7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20abee4dad939004c06936a96334fcf3
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_427b71d733d25363373bf763ddfe7240(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4276], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe2c1ade84c6e0d802bd71e4a87b12df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_427b71d733d25363373bf763ddfe7240
    def get_inputs(self):
        return [
            paddle.uniform([4276], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_bc619a684123ab2ce6f240ec4ccf97ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8732, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f98c6ab87f43e0363ef84052c0796ed2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc619a684123ab2ce6f240ec4ccf97ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_d8b36852dfb60e59fd7a3e2bec6f1ae4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_119efed712870d89531de6271b073f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b36852dfb60e59fd7a3e2bec6f1ae4
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_aa4c47a296f3a28d2d02c9cd1cf0bf55(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_33ba46db6451e5056b3cfe05e7ff01b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa4c47a296f3a28d2d02c9cd1cf0bf55
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_6c53897818ae9630e173eb9e7385db1a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 21824, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ae1afe5aec14c15a3c0a0f0ecc6f9ec6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c53897818ae9630e173eb9e7385db1a
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ae1afe5aec14c15a3c0a0f0ecc6f9ec6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c53897818ae9630e173eb9e7385db1a
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0451eeb7efec99cf33667bf194111aa6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e011829a7f713e6349778bfa6b77e261(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0451eeb7efec99cf33667bf194111aa6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.003741987282410264, 0.025496795773506165]], [[0.1733752191066742, 0.11218663305044174]], [[0.002372242044657469, 0.03849072754383087]], [[0.06514905393123627, 0.04417464882135391]], [[0.0861290991306305, 0.07314150780439377]], [[0.11290349066257477, 0.028851419687271118]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6297dcda0b3e80a3f6625cd82374e582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0451eeb7efec99cf33667bf194111aa6
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.000538684253115207, 0.17035910487174988]], [[0.21569329500198364, 0.09403947740793228]], [[0.06603813916444778, 0.003583379089832306]], [[0.004210277460515499, 0.05611496791243553]], [[0.015477688051760197, 0.015272915363311768]], [[0.0028488633688539267, 0.014013008214533329]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_12576afce880d9ada2bb60c4acaf1bf6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 16, 49, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa5db0f32cfe6a641f45b3996ac23eec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12576afce880d9ada2bb60c4acaf1bf6
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a27299759704567a560da451d04fc1bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_62e54f1a2bcf364b2a17e1c7e80947c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a27299759704567a560da451d04fc1bc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12691421806812286, 0.1352023482322693, 0.013493647798895836, 0.07988381385803223, 0.2625018060207367, 0.05280955508351326, 0.1192699745297432, 0.006763939280062914, 0.16404058039188385, 0.09374766796827316, 0.0888858288526535, 0.24979951977729797, 0.08853854984045029, 0.13602347671985626, 0.20221270620822906, 0.10543174296617508], dtype='float32').reshape([16]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_00f34fe5927f6f79f4a2a301ffe19769(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[53, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc023d94ddc994b8e214bb6b3b7858e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00f34fe5927f6f79f4a2a301ffe19769
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_0265288a369f00b23f64ce0cb9c3cd83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[150], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a72edf6f7062e960d1d78e2549136d53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0265288a369f00b23f64ce0cb9c3cd83
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_e8af63b8085dbd5c04c5bc180b410ca9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9cfd532a79835e635803c4f5871c94a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8af63b8085dbd5c04c5bc180b410ca9
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_c7926fbb1163fa47ec956abe564095b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_141074ecbf8078b6e3fc0ee5754278fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7926fbb1163fa47ec956abe564095b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_463e59c76ebdd8bcde340eb54df4488f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1758, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e0085c954cf0d8214c3b061cc66fc0c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463e59c76ebdd8bcde340eb54df4488f
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e0085c954cf0d8214c3b061cc66fc0c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463e59c76ebdd8bcde340eb54df4488f
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_323d19446faca91132f2906871d32852(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15200], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_03a90b9fbfb556fff5b3f02257335f8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_323d19446faca91132f2906871d32852
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_119efed712870d89531de6271b073f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b36852dfb60e59fd7a3e2bec6f1ae4
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_55f07d5f7a0af04d8aacb35656506d17(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d82b4bc32ef6772f560b9e7f28c9971(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55f07d5f7a0af04d8aacb35656506d17
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.25677168369293213, 0.06483365595340729, 0.1840917319059372, 0.21405082941055298], [0.131243497133255, 0.286072313785553, 0.14388522505760193, 0.09586738049983978], [0.44949236512184143, 0.0903753936290741, 0.08646504580974579, 0.10369832813739777], [0.1921076476573944, 0.2811718285083771, 0.062019556760787964, 0.44114696979522705], [0.005936041474342346, 0.09648439288139343, 0.23188406229019165, 0.2345970869064331]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_a13a99c0a281f81ea3ece857939576f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 4, 16, 49, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5108927eaafe4e90a4169107d14e58d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a13a99c0a281f81ea3ece857939576f5
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_119efed712870d89531de6271b073f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b36852dfb60e59fd7a3e2bec6f1ae4
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_83ff0a1bfc040e5b023427949576174f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55f07d5f7a0af04d8aacb35656506d17
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1816149801015854, 0.09483715891838074, 0.07430045306682587, 0.05605560541152954], [0.10881158709526062, 0.026398949325084686, 0.24707551300525665, 0.000649183988571167], [0.334237277507782, 0.32552292943000793, 0.013933449983596802, 0.08204120397567749], [0.10881158709526062, 0.026398949325084686, 0.24707551300525665, 0.000649183988571167], [0.334237277507782, 0.32552292943000793, 0.013933449983596802, 0.08204120397567749]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_05df947d6acd8be87e09b49d9cb4f6f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21824, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ecb55b5215597d8ef06cf098549e50d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05df947d6acd8be87e09b49d9cb4f6f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_80be9cf6ac97b50650b4ecac2f261d33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bdd1e00ca5b576da70c3c9152a507cd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80be9cf6ac97b50650b4ecac2f261d33
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ca45a1a7b4a9a98c4d070aac7ba5aba6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5593, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ff204ead351f208ed1d84f55b23f899d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca45a1a7b4a9a98c4d070aac7ba5aba6
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_ff204ead351f208ed1d84f55b23f899d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca45a1a7b4a9a98c4d070aac7ba5aba6
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_119efed712870d89531de6271b073f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b36852dfb60e59fd7a3e2bec6f1ae4
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_e53755050196c75d36dc898ece19ba37(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2fc5d5a21bc97f490bbdb9460d71923a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e53755050196c75d36dc898ece19ba37
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05213640630245209, 0.09683552384376526, 0.3227783441543579, 0.0005768388509750366], [0.016525864601135254, 0.354070782661438, 0.051243215799331665, 0.14045239984989166], [0.14370733499526978, 0.3452792465686798, 0.244966059923172, 0.12276658415794373], [0.016525864601135254, 0.354070782661438, 0.051243215799331665, 0.14045239984989166], [0.14370733499526978, 0.3452792465686798, 0.244966059923172, 0.12276658415794373], [0.28111082315444946, 0.2227138876914978, 0.09374728798866272, 0.29603323340415955], [0.28111082315444946, 0.2227138876914978, 0.09374728798866272, 0.29603323340415955]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_4c47400cfd643ef4718c667219c56e10(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_93891f8ed345daad803ec43ed07d7600(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c47400cfd643ef4718c667219c56e10
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_40176fa4cd7b5ee67634a8116ee540a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 32, 16, 49, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a3bb4bb692b9bd975b222956fe51a34b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40176fa4cd7b5ee67634a8116ee540a0
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_119efed712870d89531de6271b073f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b36852dfb60e59fd7a3e2bec6f1ae4
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_4c717fa440ef63d40486c4fc886aa005(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[103, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ba785d23ab753217bfa0c931abc5d29e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c717fa440ef63d40486c4fc886aa005
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_141074ecbf8078b6e3fc0ee5754278fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7926fbb1163fa47ec956abe564095b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_638a0696aeb404f135489745135b6a67(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1763, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab9377a40b26344ef940b56cd436464f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_638a0696aeb404f135489745135b6a67
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_ab9377a40b26344ef940b56cd436464f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_638a0696aeb404f135489745135b6a67
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_03a90b9fbfb556fff5b3f02257335f8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_323d19446faca91132f2906871d32852
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_a8f2ed709d5ede07eba99d8ceed79343(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 8, 16, 49, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_78042a8335fac2d60e959b1fa5ceb81d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8f2ed709d5ede07eba99d8ceed79343
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d9f1887748efc34607438842a50f1e5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e6357c94add2909903dd2d10c60f5028(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9f1887748efc34607438842a50f1e5c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0632820650935173, 0.2012043446302414, 0.09819939732551575, 0.19736462831497192, 0.10661789029836655, 0.15769129991531372, 0.18587596714496613, 0.033052314072847366, 0.09060952812433243, 0.21144439280033112, 0.005274119321256876, 0.009855720214545727, 0.1895647794008255, 0.1720527708530426, 0.10415849834680557, 0.184909850358963, 0.24222317337989807, 0.05828609690070152, 0.06454712152481079, 0.021931804716587067, 0.05520737171173096, 0.1563003957271576, 0.13837598264217377, 0.10057298839092255], dtype='float32').reshape([24]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_9ca35e222902e313f9424c2b95a74c37(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b76c6be6dfa6375c57f43c5a2785891(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ca35e222902e313f9424c2b95a74c37
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ebed60b074e52e5c0f2b24cfd1199669(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1490, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e810078b7e68dbc76ae8960c42336135(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebed60b074e52e5c0f2b24cfd1199669
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e810078b7e68dbc76ae8960c42336135(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebed60b074e52e5c0f2b24cfd1199669
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_bda2ffddc8328c6339fde85cf3f726f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 16, 16, 49, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a4d03a62f9229a45701647425d899c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bda2ffddc8328c6339fde85cf3f726f8
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ea7831c0d6050d1b61b4dd4452922e1d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8cff2e08b6cbf8b24e629fc2a0433c34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea7831c0d6050d1b61b4dd4452922e1d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.099599190056324, 0.18784429132938385, 0.205540269613266, 0.19256934523582458], dtype='float32').reshape([4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_119efed712870d89531de6271b073f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b36852dfb60e59fd7a3e2bec6f1ae4
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_e22156f7dfc89bef762781ebe21a313a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_194a5a0316c97ecd63343f81e62e582b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e22156f7dfc89bef762781ebe21a313a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4721890091896057, 0.13927704095840454, 0.32948899269104004, 0.10651195794343948], [0.0006179213523864746, 0.0033878684043884277, 0.0574188232421875, 0.054265450686216354], [0.3772001266479492, 0.07573233544826508, 0.05967779457569122, 0.010373055934906006], [0.22724922001361847, 0.15835122764110565, 0.28487300872802734, 0.1891089379787445], [0.22724922001361847, 0.15835122764110565, 0.28487300872802734, 0.1891089379787445], [0.3772001266479492, 0.07573233544826508, 0.05967779457569122, 0.010373055934906006]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_119efed712870d89531de6271b073f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b36852dfb60e59fd7a3e2bec6f1ae4
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_f923448068114d13ce1d30c1efe9293f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55f07d5f7a0af04d8aacb35656506d17
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0917310118675232, 0.15178906917572021, 0.03986068069934845, 0.002778302878141403], [0.25007253885269165, 0.07839547097682953, 0.16669723391532898, 0.34016889333724976], [0.037536993622779846, 0.22154566645622253, 0.2786290645599365, 0.3307667374610901], [0.1383073329925537, 0.31077325344085693, 0.33028000593185425, 0.13818231225013733], [0.0917310118675232, 0.15178906917572021, 0.03986068069934845, 0.002778302878141403]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_119efed712870d89531de6271b073f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b36852dfb60e59fd7a3e2bec6f1ae4
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_b86d86a612f58a5c07da518398543e0c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0560bb38eb7bb60a23f947c363a33d7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b86d86a612f58a5c07da518398543e0c
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_3a4d03a62f9229a45701647425d899c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bda2ffddc8328c6339fde85cf3f726f8
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_119efed712870d89531de6271b073f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b36852dfb60e59fd7a3e2bec6f1ae4
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_b3278e2fe9524238282340b6eec9e51f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d4a9bce564390bf6fa0b653cb75c150d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3278e2fe9524238282340b6eec9e51f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2753981351852417, 0.27793630957603455, 0.17279444634914398, 0.1332867443561554], [0.09361551702022552, 0.1447596698999405, 0.08765412867069244, 0.32762980461120605], [0.17097410559654236, 0.2685585618019104, 0.0018512457609176636, 0.31080305576324463], [0.12934653460979462, 0.0022540315985679626, 0.2781771421432495, 0.30496832728385925]], dtype='float32').reshape([4, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_0ac0f08df411b49b5e74c6c02dc9a9dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 19, 34], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_524952eca2a67799b5d929e872e3b3ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ac0f08df411b49b5e74c6c02dc9a9dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_78042a8335fac2d60e959b1fa5ceb81d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8f2ed709d5ede07eba99d8ceed79343
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_119efed712870d89531de6271b073f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b36852dfb60e59fd7a3e2bec6f1ae4
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_f754b4b075ae6f9655c86df7aac156b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[84, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2115c44767c82b9d3e1dce3c114c6fc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f754b4b075ae6f9655c86df7aac156b9
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_47a0ac4d198e0b8c0fe99ccee52434c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[950], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eddcc80240b972e1849bde664ec12b1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47a0ac4d198e0b8c0fe99ccee52434c5
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_bd60ccf92575e1c65b4427f0c7679788(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8816], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9dcb63321b3db94db003be885985c8c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd60ccf92575e1c65b4427f0c7679788
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_38252a922e4c6735563af4ed4ec4a1e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2f1393721b927b7a3b8edd70b18a7601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38252a922e4c6735563af4ed4ec4a1e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_25cb2f3311ff38a764d3a5ca59aef236(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2076, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_82fddaa8b82d1eed54e5b0b280ecc130(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25cb2f3311ff38a764d3a5ca59aef236
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_82fddaa8b82d1eed54e5b0b280ecc130(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25cb2f3311ff38a764d3a5ca59aef236
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_4f9018b41f973066df108823e995a6b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 152, 272], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c4e018f3c212751d5c10e7ca137e4efe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f9018b41f973066df108823e995a6b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_119efed712870d89531de6271b073f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b36852dfb60e59fd7a3e2bec6f1ae4
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_bbf0105f46405eb12cec1fad599afb78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e53755050196c75d36dc898ece19ba37
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03508669137954712, 0.16850346326828003, 0.18960177898406982, 0.007154777646064758], [0.03508669137954712, 0.16850346326828003, 0.18960177898406982, 0.007154777646064758], [0.10633885860443115, 0.07276816666126251, 0.029908359050750732, 0.12192053347826004], [0.06507094949483871, 0.243668332695961, 0.21072156727313995, 0.08438259363174438], [0.11407667398452759, 0.0010586678981781006, 0.3051604628562927, 0.2620214521884918], [0.06873470544815063, 0.0030619800090789795, 0.05197077989578247, 0.09565383195877075], [0.26225602626800537, 0.06410759687423706, 0.13850826025009155, 0.04361702501773834]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_8a292a89346751d09e65408be02e49ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e0b5ad696d99aa8f4b17f98e1c17438f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a292a89346751d09e65408be02e49ba
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_1c5a80eba96c17d94dfab922aab8264a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4642, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_750221f01886831b4c96a86594a2b161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c5a80eba96c17d94dfab922aab8264a
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_750221f01886831b4c96a86594a2b161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c5a80eba96c17d94dfab922aab8264a
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_19769968e656452c000b085cdaf11bbc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4903], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a5854cc59037cc6fe155bff3e7f7dfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19769968e656452c000b085cdaf11bbc
    def get_inputs(self):
        return [
            paddle.uniform([4903], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_4d5fe048ba1e9650b74e1e62925b2319(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1230], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f81786dd6f522276dace1d33b542294c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d5fe048ba1e9650b74e1e62925b2319
    def get_inputs(self):
        return [
            paddle.uniform([1230], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_0351607623847b7f1c3f1133e652ee0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2434, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_128ad9b1a90132c3eed540edae3d8e73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0351607623847b7f1c3f1133e652ee0b
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_153a5d2a5f4c9a909164b52fedf95e1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3dd9bd1207c4ac31ebb5d565f0a1e505(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_153a5d2a5f4c9a909164b52fedf95e1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_38968869ae1447b3dd9873eb9923dbac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1047, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e57896be304044219de3ce3ac7b6bc50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38968869ae1447b3dd9873eb9923dbac
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e57896be304044219de3ce3ac7b6bc50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38968869ae1447b3dd9873eb9923dbac
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_119efed712870d89531de6271b073f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b36852dfb60e59fd7a3e2bec6f1ae4
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_52c9cb3fa1eba4f9388708a751d2ed7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e22156f7dfc89bef762781ebe21a313a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19283601641654968, 0.44643473625183105, 0.1266147792339325, 0.3170720636844635], [0.054931044578552246, 0.28975534439086914, 0.2113265097141266, 0.08645415306091309], [0.054931044578552246, 0.28975534439086914, 0.2113265097141266, 0.08645415306091309], [0.3403587341308594, 0.34647834300994873, 0.08163797855377197, 0.00804758071899414], [0.24778346717357635, 0.10596618056297302, 0.31448638439178467, 0.1395723819732666], [0.38121309876441956, 0.2610614597797394, 0.1390228420495987, 0.18141451478004456]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_1ba897b1a7eb20fe914cff1ff8626b74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 2, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d660e2014a1dd1e2d563de4e8dd9329(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ba897b1a7eb20fe914cff1ff8626b74
    def get_inputs(self):
        return [
            paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_00d7d9283dd4c29fb1dd1d6d8613fda8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 32, 16, 49, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c238daee247777b34740bb92d90c79fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00d7d9283dd4c29fb1dd1d6d8613fda8
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e62e0d3972589748c949d185b1fbb896(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 2, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d33f54e92e4baba4191392463d76bb82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62e0d3972589748c949d185b1fbb896
    def get_inputs(self):
        return [
            paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e00148bf9f8335400b542cc1411a67e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92043dac2eeeff33c1cece22107618ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e00148bf9f8335400b542cc1411a67e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_70b686a24bfe32009d96c3db961e3af8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2359, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_28ce571c5113bbf51b064915b3217a7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70b686a24bfe32009d96c3db961e3af8
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_28ce571c5113bbf51b064915b3217a7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70b686a24bfe32009d96c3db961e3af8
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_62774cf9c66ed8e892c0230584957966(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a733dc1ccde02f5cdb260b788935f9ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62774cf9c66ed8e892c0230584957966
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_21fe0c6eb4ff783e2da3179883ee1d05(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3049, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f7740222f624192070dc9eeb5c0c386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21fe0c6eb4ff783e2da3179883ee1d05
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_6f7740222f624192070dc9eeb5c0c386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21fe0c6eb4ff783e2da3179883ee1d05
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_7aeedbfea6f22ce056568b76a1c0e99d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6bd8b78fa5423946a448d5e4e098dc89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7aeedbfea6f22ce056568b76a1c0e99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_51db62257f8d60a2df4219b6e0083d33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3806, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_19a02632a26df33609e43bab92fdbb69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51db62257f8d60a2df4219b6e0083d33
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_19a02632a26df33609e43bab92fdbb69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51db62257f8d60a2df4219b6e0083d33
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_368c13dc53af6c29bc0023a310da4a03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[247], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a77d6825c22393b731920daf3a8f3034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_368c13dc53af6c29bc0023a310da4a03
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_02aafe989cc06cb2fa83f529a56fa316(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 16, 49, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_98db00416fc3680508f1abb1ccac5633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02aafe989cc06cb2fa83f529a56fa316
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_04ad5e1ffe49dde6ff94535f3e833f23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 16, 49, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_95a99ad1f21040514e8a1a261f84308f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04ad5e1ffe49dde6ff94535f3e833f23
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a3bb4bb692b9bd975b222956fe51a34b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40176fa4cd7b5ee67634a8116ee540a0
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_815961350f8c7c1562fc0d786a813530(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4db023197ae4445ed51ac9141f43d2ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_815961350f8c7c1562fc0d786a813530
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1740274578332901, 0.2572338581085205, 0.2279798984527588, 0.10160446166992188, 0.16043861210346222, 0.23984308540821075, 0.04370877891778946, 0.27654924988746643, 0.04856458306312561, 0.003998385742306709, 0.10652326792478561, 0.16278232634067535, 0.12810862064361572, 0.14386767148971558, 0.1635606437921524, 0.15939965844154358, 0.20407362282276154, 0.20692703127861023, 0.07404422014951706, 0.01399579644203186], dtype='float32').reshape([20]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_4a9ae31176432e6aa9afd4cd2366a9f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17370], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_541ea6f23664ca8c90751824c1c63219(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a9ae31176432e6aa9afd4cd2366a9f8
    def get_inputs(self):
        return [
            paddle.uniform([17370], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_98db00416fc3680508f1abb1ccac5633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02aafe989cc06cb2fa83f529a56fa316
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f44421e6662cea73d5b5bb9c640c9eb2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cb18840f98e4786cf115b70b3261aecc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f44421e6662cea73d5b5bb9c640c9eb2
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_0ae17d630b00a9a62b9882afd471334e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[47, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_149c1643750e2430ac11413df97831ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ae17d630b00a9a62b9882afd471334e
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_31f79c91572dc9c6dcdd6ddd81ae954a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_86b6aa7e8a1333f41d60843bbde280ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31f79c91572dc9c6dcdd6ddd81ae954a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_be3cf52dac08e07ccfc56b29736ddb8b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2054, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ca4260d878f3cde32d157edbddb3b94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be3cf52dac08e07ccfc56b29736ddb8b
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_7ca4260d878f3cde32d157edbddb3b94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be3cf52dac08e07ccfc56b29736ddb8b
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_f749254b81779169df24a356dd1d2bca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 2, 16, 9, 112, 112], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eac7996d4b3049591acc87002b649b40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f749254b81779169df24a356dd1d2bca
    def get_inputs(self):
        return [
            paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_01ab21c0a12f87c18055da17b6a48ff8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[551], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b225527f6e98de3d9d3f59315f7f5938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01ab21c0a12f87c18055da17b6a48ff8
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_119efed712870d89531de6271b073f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b36852dfb60e59fd7a3e2bec6f1ae4
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_735ce455a1277b751447f881f1ebce9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55f07d5f7a0af04d8aacb35656506d17
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.00888514518737793, 0.04896102473139763, 0.0483778715133667, 0.03471830487251282], [0.009788215160369873, 0.05413275957107544, 0.30535221099853516, 0.07475248724222183], [0.39217886328697205, 0.2077333927154541, 0.19593361020088196, 0.054433196783065796], [0.39217886328697205, 0.2077333927154541, 0.19593361020088196, 0.054433196783065796], [0.09205174446105957, 0.13316354155540466, 0.10884398221969604, 0.3186963200569153]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_c238daee247777b34740bb92d90c79fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00d7d9283dd4c29fb1dd1d6d8613fda8
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_46d07dcb40220ee154e558100391e528(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3800], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0289717df0ee15ed6221df2e49e6c6a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46d07dcb40220ee154e558100391e528
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_d3cac74578dc0dac8be74d0a72896345(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2204], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c2e0846b16c5eca90bf5b489cf3faa90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3cac74578dc0dac8be74d0a72896345
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_2595cc2ed6fda1a8397cce454a9e5727(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a9fbc12f5a3277e5d589c07a45c5edac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2595cc2ed6fda1a8397cce454a9e5727
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_c8e2299a28c41b31e06c101878243986(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_67b0f93271c594e24ac392e2caf6dc00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e2299a28c41b31e06c101878243986
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7c852f37a5067f4bf484c73907389824(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4218, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a28ffe0cc32c2036f4f3467f97830290(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c852f37a5067f4bf484c73907389824
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_a28ffe0cc32c2036f4f3467f97830290(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c852f37a5067f4bf484c73907389824
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_119efed712870d89531de6271b073f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b36852dfb60e59fd7a3e2bec6f1ae4
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_8ceb65385ee94164e1e29d49152f49ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e53755050196c75d36dc898ece19ba37
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.01474866271018982, 0.24872198700904846, 0.05770754814147949, 0.18724988400936127], [0.09170161187648773, 0.045166343450546265, 0.006407499313354492, 0.06573697924613953], [0.20618627965450287, 0.32045453786849976, 0.020322144031524658, 0.06753784418106079], [0.01474866271018982, 0.24872198700904846, 0.05770754814147949, 0.18724988400936127], [0.2491391897201538, 0.22439298033714294, 0.35869306325912476, 0.05700208991765976], [0.017012417316436768, 0.05802673101425171, 0.03807038068771362, 0.10305461287498474], [0.2491391897201538, 0.22439298033714294, 0.35869306325912476, 0.05700208991765976]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_8a461dfc55d1a85428f2fcf6addf8ecd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[52, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_51374365290f434d6d182ed5389fc1d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a461dfc55d1a85428f2fcf6addf8ecd
    def get_inputs(self):
        return [
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_95a99ad1f21040514e8a1a261f84308f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04ad5e1ffe49dde6ff94535f3e833f23
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()