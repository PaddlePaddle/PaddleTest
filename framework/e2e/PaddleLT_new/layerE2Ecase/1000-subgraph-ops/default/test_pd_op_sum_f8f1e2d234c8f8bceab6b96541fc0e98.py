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


class PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9bc2d894f33a1a615d8eaac6d0290a10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([4460], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_1df5fe60ec9c97e97c524ee9f1a94e3a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_027faa618bdf941e7b2d8bcea1f160d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df5fe60ec9c97e97c524ee9f1a94e3a
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e561f0e58ce10051a22e36f19791535d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_886b743927c2c8972c1588353a0593e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_60a2efd0575acd1a04c4a5efbf29b092(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_886b743927c2c8972c1588353a0593e8
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_17459e2632aaff2b7c1eb65680bfe5fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1cc06b95cde240df36372a4ab4aaf8b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17459e2632aaff2b7c1eb65680bfe5fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1cc06b95cde240df36372a4ab4aaf8b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17459e2632aaff2b7c1eb65680bfe5fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5dcf35d4df812b463d776ba336743170(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17459e2632aaff2b7c1eb65680bfe5fb
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.012445604428648949, 0.03725549578666687]], [[0.005800215993076563, 0.011232854798436165]], [[0.23404957354068756, 0.005940473638474941]], [[0.033425621688365936, 0.018772045150399208]], [[0.12282110005617142, 0.22190912067890167]], [[0.009733823128044605, 0.03293434903025627]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f6f5fe369e911d93120a90374db259a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17459e2632aaff2b7c1eb65680bfe5fb
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0026597813703119755, 0.06375440955162048]], [[0.008435319177806377, 2.640220736793708e-05]], [[0.14170944690704346, 0.011285020038485527]], [[0.08177555352449417, 0.056741148233413696]], [[0.005669548176229, 0.16143694519996643]], [[0.1153835654258728, 0.08475687354803085]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class TestPrimitiveOp_0bad41ad73c4d7aad19363fafe12a77b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09749365597963333, 0.20418159663677216, 0.1878623068332672, 0.13012243807315826, 0.1662166565656662, 0.06850031763315201, 0.0015883890446275473, 0.020663680508732796, 0.052616044878959656, 0.2103569209575653, 0.21663494408130646, 0.12605345249176025, 0.18519194424152374, 0.1061423197388649, 0.1676478385925293, 0.05291515216231346], dtype='float32').reshape([16]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_31e912bd392d2cf5e4557f81740bd153(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_886b743927c2c8972c1588353a0593e8
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_af4827c4efde6869d61cc4870ea458d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_1b46b465b995c7fd58d5c2a0a15e3964(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_9dcb03aaa245a2305d4e185af84aaab9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a7049b47eb3890422ddf0ed6998ec7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dcb03aaa245a2305d4e185af84aaab9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_0cd936b40a76d2b176986b4396644121(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6885e68c6bd8d52cbb20efb868994d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_d6885e68c6bd8d52cbb20efb868994d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_112aec5563db8fa60e1b101bafd16652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e561f0e58ce10051a22e36f19791535d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_24e9f79abaf398585e0b93a1e280fec5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_886b743927c2c8972c1588353a0593e8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06600350141525269, 0.12325924634933472, 0.14050322771072388, 0.09338384866714478], [0.2529301643371582, 0.19529643654823303, 0.035754598677158356, 0.3732248842716217], [0.05762083828449249, 0.33048099279403687, 0.3098161816596985, 0.046964749693870544], [0.1361105740070343, 0.39801257848739624, 0.07723602652549744, 0.10689233243465424], [0.21829457581043243, 0.25948721170425415, 0.19501274824142456, 0.08205263316631317]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_e561f0e58ce10051a22e36f19791535d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_818bf585e77c2345939a0da4c713c1c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_886b743927c2c8972c1588353a0593e8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.29332220554351807, 0.17943236231803894, 0.0017442405223846436, 0.10830046236515045], [0.04921819269657135, 0.21070674061775208, 0.37740403413772583, 0.13908714056015015], [0.02563472092151642, 0.16697435081005096, 0.006776377558708191, 0.11909252405166626], [0.04921819269657135, 0.21070674061775208, 0.37740403413772583, 0.13908714056015015], [0.02563472092151642, 0.16697435081005096, 0.006776377558708191, 0.11909252405166626]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e74962f5392b865fab4650bb615f1237(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df5fe60ec9c97e97c524ee9f1a94e3a
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_f06f79bbd6ff748e18a7a0f70fe8877d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dcb03aaa245a2305d4e185af84aaab9
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_245017b3b1b54f829fceda6efb3f7129(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_245017b3b1b54f829fceda6efb3f7129(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e561f0e58ce10051a22e36f19791535d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_751e2e39014efedad314400d0c089504(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_886b743927c2c8972c1588353a0593e8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.42621099948883057, 0.1255248486995697, 0.13389798998832703, 0.25021958351135254], [0.09774552285671234, 0.2969127297401428, 0.33496904373168945, 0.39779192209243774], [0.06356087327003479, 0.019604787230491638, 0.08770968019962311, 0.3853555917739868], [0.09774552285671234, 0.2969127297401428, 0.33496904373168945, 0.39779192209243774], [0.06356087327003479, 0.019604787230491638, 0.08770968019962311, 0.3853555917739868], [0.10860182344913483, 0.15528208017349243, 0.3284634053707123, 0.05508685111999512], [0.10860182344913483, 0.15528208017349243, 0.3284634053707123, 0.05508685111999512]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_81dad9eeb6563002b10c488f1bedaf2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
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


class TestPrimitiveOp_e561f0e58ce10051a22e36f19791535d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_d0e8ea1a9d928bfa4bead763cc154115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_886b743927c2c8972c1588353a0593e8
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_1a7049b47eb3890422ddf0ed6998ec7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dcb03aaa245a2305d4e185af84aaab9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_28093826e210c82dabb65e5e99d11b41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_28093826e210c82dabb65e5e99d11b41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_112aec5563db8fa60e1b101bafd16652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
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


class TestPrimitiveOp_56ac42b23b057d646500269a7ae994bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.022885655984282494, 0.16699419915676117, 0.10250034928321838, 0.05291581526398659, 0.24314573407173157, 0.01018413808196783, 0.07149253040552139, 0.21883918344974518, 0.05653005838394165, 0.23525764048099518, 0.24263663589954376, 0.035404372960329056, 0.00032842261134646833, 0.24848103523254395, 0.22433628141880035, 0.10429609566926956, 0.0046191769652068615, 0.18702290952205658, 0.09567490220069885, 0.22300565242767334, 0.0429198332130909, 0.1674208641052246, 0.15969066321849823, 0.2024993598461151], dtype='float32').reshape([24]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_abf66c82bad00d4ca5868a400cc784cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dcb03aaa245a2305d4e185af84aaab9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_be706cc8ab2b45fbfc5525f144132167(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_be706cc8ab2b45fbfc5525f144132167(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_4ecad889f5887ef87e4832658f005252(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10500867664813995, 0.17738689482212067, 0.10432569682598114, 0.24906568229198456], dtype='float32').reshape([4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e561f0e58ce10051a22e36f19791535d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_adea85d9ef41a79548f316e20b5e31ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_886b743927c2c8972c1588353a0593e8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0347633957862854, 0.2688142955303192, 0.15619760751724243, 0.12203063070774078], [0.04207935929298401, 0.25613322854042053, 0.023173600435256958, 0.18292436003684998], [0.06694593280553818, 0.13091182708740234, 0.36519718170166016, 0.13943824172019958], [0.054342061281204224, 0.1302337646484375, 0.2963424324989319, 0.08765411376953125], [0.054342061281204224, 0.1302337646484375, 0.2963424324989319, 0.08765411376953125], [0.06694593280553818, 0.13091182708740234, 0.36519718170166016, 0.13943824172019958]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e561f0e58ce10051a22e36f19791535d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_94faf162bc46b7d8b66febc963102c40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_886b743927c2c8972c1588353a0593e8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16947636008262634, 0.12996846437454224, 0.0761517733335495, 0.06142064929008484], [0.010005325078964233, 0.010751783847808838, 0.16052129864692688, 0.20979206264019012], [0.3373373746871948, 0.017422378063201904, 0.02375468611717224, 0.03213486075401306], [0.3443434238433838, 0.40481501817703247, 0.048192352056503296, 0.11602441966533661], [0.16947636008262634, 0.12996846437454224, 0.0761517733335495, 0.06142064929008484]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e561f0e58ce10051a22e36f19791535d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_3dce8e356ff53f7e6a9f6e227e4ee7e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_886b743927c2c8972c1588353a0593e8
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


class TestPrimitiveOp_e561f0e58ce10051a22e36f19791535d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_68d954e2b0349e5ba3fdd03b20b87e61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_886b743927c2c8972c1588353a0593e8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12086917459964752, 0.08581306040287018, 0.04574468731880188, 0.38157597184181213], [0.4106341600418091, 0.02336856722831726, 0.06159588694572449, 0.03985142707824707], [0.2122405618429184, 0.05072645843029022, 0.11940930038690567, 0.0011862218379974365], [0.2085941582918167, 0.21888814866542816, 0.11163263022899628, 0.31563934683799744]], dtype='float32').reshape([4, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_1ecc50182608e55e639792d63838cc6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0c5666aca3e5990eee4af62155689704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ecc50182608e55e639792d63838cc6f
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


class TestPrimitiveOp_e561f0e58ce10051a22e36f19791535d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_6fd2bcc394207f046ef82f4a5e8b6b6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_886b743927c2c8972c1588353a0593e8
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_bbd7afd4595d15638d08e1355fe9cfdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_28983d559b3601a95deadef96d40c8a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_f42d63bfa6d292b0c890ad2b9bab0119(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dcb03aaa245a2305d4e185af84aaab9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d220bfc0d779e6103f4f5c07d0d9227c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_d220bfc0d779e6103f4f5c07d0d9227c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_1075a0030e4aacf2a395275493f492a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ecc50182608e55e639792d63838cc6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_e561f0e58ce10051a22e36f19791535d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_a440d34c5691bc92d24115ea5fd594bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_886b743927c2c8972c1588353a0593e8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06793880462646484, 0.1796111911535263, 0.41419026255607605, 0.23429150879383087], [0.06793880462646484, 0.1796111911535263, 0.41419026255607605, 0.23429150879383087], [0.20293423533439636, 0.042559683322906494, 0.03312063217163086, 0.017667576670646667], [0.39120301604270935, 0.03000320866703987, 0.09842769801616669, 0.34845295548439026], [0.12772080302238464, 0.18704058229923248, 0.3924790620803833, 0.21641282737255096], [0.36120250821113586, 0.09090563654899597, 0.06719005107879639, 0.23598814010620117], [0.01706467568874359, 0.13718575239181519, 0.23035810887813568, 0.017347488552331924]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_969f7d9b04c1ce51f0e057ece6f6a82d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dcb03aaa245a2305d4e185af84aaab9
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ad640aa31d12315c69f2be66e45a96f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_ad640aa31d12315c69f2be66e45a96f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_6a20f265857ed952778bf3e46c857720(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([4942], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_44e78f2123397715b31bf317a90c5931(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([1206], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_3a73c0df0af43bb181db2e918dc16902(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df5fe60ec9c97e97c524ee9f1a94e3a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_189316a4b0e2cc47466135ef69376daf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dcb03aaa245a2305d4e185af84aaab9
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_722647875131ed429aea08d0dd7341f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_722647875131ed429aea08d0dd7341f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e561f0e58ce10051a22e36f19791535d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e7d84d97da12030e952a7727bb9ef982(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_886b743927c2c8972c1588353a0593e8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0011395514011383057, 0.3579179644584656, 0.19031499326229095, 0.3789938688278198], [0.14381445944309235, 0.26047101616859436, 0.28549501299858093, 0.2758251428604126], [0.14381445944309235, 0.26047101616859436, 0.28549501299858093, 0.2758251428604126], [0.1933608502149582, 0.1633141189813614, 0.14810127019882202, 0.1474897861480713], [0.06986059248447418, 0.08585286140441895, 0.023728124797344208, 0.2839673161506653], [0.02392372488975525, 0.06696397066116333, 0.20791184902191162, 0.3662828803062439]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_9ceb40b3d453e66617aa30369a9e85c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e553d1d8e285dc43e4f1ff5c2140f0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ceb40b3d453e66617aa30369a9e85c6
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


class PrimitiveOp_86aa246f32976f211d2ce6985d7c5ef4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e89feb02defce54bb44e7591f83dd0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86aa246f32976f211d2ce6985d7c5ef4
    def get_inputs(self):
        return [
            paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_514a49077a718e200826064fed431e52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dcb03aaa245a2305d4e185af84aaab9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_305ee6c95d642f37536c3afeebcfa9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_305ee6c95d642f37536c3afeebcfa9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_46009e9e1fcf22a04cd9c9a5db6537c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dcb03aaa245a2305d4e185af84aaab9
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_920b6e27585b4cf984d0c59ca47e4501(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_920b6e27585b4cf984d0c59ca47e4501(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_30c2c6e06d4a1a0f0b5a07b3adf1a330(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dcb03aaa245a2305d4e185af84aaab9
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3608c19c5d2b6bf6904b27233f82c9b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_3608c19c5d2b6bf6904b27233f82c9b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e1609278c26928e2f5e515a242259732(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
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


class TestPrimitiveOp_b5ac551343de4133ea03159ef968a3ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12528188526630402, 0.13080038130283356, 0.01519912201911211, 0.09477114677429199, 0.0759349912405014, 0.0996357873082161, 0.22864927351474762, 0.1952035278081894, 0.17059211432933807, 0.12125129997730255, 0.0819561630487442, 0.007883278653025627, 0.23232892155647278, 0.11551108211278915, 0.24835118651390076, 0.004859336651861668, 0.1575036197900772, 0.2514082193374634, 0.12115499377250671, 0.08058024942874908], dtype='float32').reshape([20]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_1c5a5168f2159a2407013573fe5b8fc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([17604], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_a7c8c63131cd6046199abf98b4d2a54f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e4cc8d43a7f6aab3e1ac351c799c7767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_886b743927c2c8972c1588353a0593e8
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_c4ba882544d04fe14f2c444d1cd22e3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dcb03aaa245a2305d4e185af84aaab9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_82d318ee649034ddaaf4f87de4a007e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_82d318ee649034ddaaf4f87de4a007e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_69088d6f409fd55fc94cbaabe7930fac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e561f0e58ce10051a22e36f19791535d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_91648586d758c919c113e9e9ed79c45a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_886b743927c2c8972c1588353a0593e8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22915200889110565, 0.006040988024324179, 0.13718536496162415, 0.06582275778055191], [0.00850994884967804, 0.03199875354766846, 0.10096690058708191, 0.31683748960494995], [0.03647342324256897, 0.18587401509284973, 0.20138487219810486, 0.0098896324634552], [0.03647342324256897, 0.18587401509284973, 0.20138487219810486, 0.0098896324634552], [0.025345072150230408, 0.1740993857383728, 0.031156376004219055, 0.023688942193984985]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_a4a9bddeb58188e7aded37fd06e7171d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_eb468a7f97c62958d988e620d9f93ca8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_aa10aec8b265ab8f93ab636878d2f392(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_886b743927c2c8972c1588353a0593e8
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_b6284933415b026283f82ee66f9f1b9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dcb03aaa245a2305d4e185af84aaab9
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c09e555a18da8c5f2bed31c9e1e0d002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_c09e555a18da8c5f2bed31c9e1e0d002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd936b40a76d2b176986b4396644121
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e561f0e58ce10051a22e36f19791535d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2948bd78c39d6c84fd1cdaab4f52b3
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_3ce0267141b497b2f351f1ba94ea99c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_886b743927c2c8972c1588353a0593e8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17515462636947632, 0.22402986884117126, 0.05804827809333801, 0.09627231955528259], [0.40905141830444336, 0.17959129810333252, 0.16124853491783142, 0.036101073026657104], [0.029448404908180237, 0.06972748041152954, 0.19806984066963196, 0.2812197506427765], [0.17515462636947632, 0.22402986884117126, 0.05804827809333801, 0.09627231955528259], [0.11691299825906754, 0.40774694085121155, 0.08399210125207901, 0.022679775953292847], [0.03554117679595947, 0.0161103755235672, 0.1517069935798645, 0.21505869925022125], [0.11691299825906754, 0.40774694085121155, 0.08399210125207901, 0.022679775953292847]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_b2c2929a6657a7a32b87b60d44e15090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_886b743927c2c8972c1588353a0593e8
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