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



class PrimitiveOp_94de098bf59f51bc5f318e663b40eef7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ea993b0dbd65608dcb16a62cb309aed8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a371de56f980181327a77e57389669bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9a6617ec456c4eab1368d97020c8e3c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([4323], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_ced0d018def9586f4026d70772d35867(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_870d9d884dfd7e807a591b409225e8ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_016df1ebed499528eeefbc832767faf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_247e5da7c073f897d47195caf3b4e04a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_2964f274495c2aa234a5be8c37975879(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9cecf4661ca63f3e8d64715406b3271d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2964f274495c2aa234a5be8c37975879
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9cecf4661ca63f3e8d64715406b3271d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2964f274495c2aa234a5be8c37975879
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3e9f067807d506199c9f5b84e077d606(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2964f274495c2aa234a5be8c37975879
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0008215791895054281, 0.02297096513211727]], [[0.001509505556896329, 0.0008934923098422587]], [[0.007353345863521099, 0.19679111242294312]], [[0.003992951475083828, 0.044404175132513046]], [[0.08140164613723755, 0.02453409507870674]], [[0.044278863817453384, 0.02057296223938465]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ed9f2e18d03da2548eb2eedeb47b001e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2964f274495c2aa234a5be8c37975879
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.006566960364580154, 0.09187708050012589]], [[0.025543373078107834, 0.09933386743068695]], [[0.0880783423781395, 0.15954287350177765]], [[0.0024919670540839434, 0.016344036906957626]], [[0.0736781656742096, 0.002479496179148555]], [[0.07606089860200882, 0.0005468514282256365]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_70a39e6ebfd0d98dcd2b757cbc94c777(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c8f886798cef705082dfae303e17ee76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2520371079444885, 0.10197052359580994, 0.24853019416332245, 0.20063623785972595, 0.05700432509183884, 0.1988341212272644, 0.17109738290309906, 0.002776582259684801, 0.007792500779032707, 0.21832634508609772, 0.10031788051128387, 0.232047438621521, 0.09687773883342743, 0.09597375243902206, 0.15565578639507294, 0.03064071573317051], dtype='float32').reshape([16]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_5da5771974ad59319f83851722439436(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e5ad10b8db401cbc38cdf0f29b60621f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_a236e4139acfba5615853faf528ad537(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_740f16aec91797da3e051acb5494e125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_685a68088ec8536d19f12fe81f918952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_685a68088ec8536d19f12fe81f918952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([1787, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_bc88f57628a852d4c30cbf50b6212390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_016df1ebed499528eeefbc832767faf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_34c7b919ef6949c2820d68586ff2f0c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08987224102020264, 0.09446462988853455, 0.0645132064819336, 0.09983265399932861], [0.09016704559326172, 0.1352386474609375, 0.2301655262708664, 0.1461450159549713], [0.21332000195980072, 0.036309823393821716, 0.010198406875133514, 0.40485113859176636], [0.1312466859817505, 0.13634832203388214, 0.39518219232559204, 0.09201464056968689], [0.06631225347518921, 0.054139167070388794, 0.10006687045097351, 0.3483027219772339]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_52f02d64ddebf187ca610a05092ad84f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_016df1ebed499528eeefbc832767faf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_c2060352030c5e1a4726e561d8d0792c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15815213322639465, 0.08740302920341492, 0.0651894211769104, 0.28641700744628906], [0.016894638538360596, 0.2502749264240265, 0.12896551191806793, 0.2630677819252014], [0.23832650482654572, 0.2584977149963379, 0.0639914721250534, 0.21149897575378418], [0.016894638538360596, 0.2502749264240265, 0.12896551191806793, 0.2630677819252014], [0.23832650482654572, 0.2584977149963379, 0.0639914721250534, 0.21149897575378418]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_6c41bbc24a31a722f7b3ea5f2bb4f9bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_b73776efdecda198010c919ce4e1d0c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ba4c7529068d44ed2a0705a938507f15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([5524, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_ba4c7529068d44ed2a0705a938507f15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([5524, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_016df1ebed499528eeefbc832767faf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_79795be6cfe174ddb5829a1d302c1e86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08980125188827515, 0.10924512147903442, 0.013537660241127014, 0.30076491832733154], [0.14479558169841766, 0.2556632459163666, 0.11069628596305847, 0.25762078166007996], [0.18983790278434753, 0.15936079621315002, 0.4752041697502136, 0.008670181035995483], [0.14479558169841766, 0.2556632459163666, 0.11069628596305847, 0.25762078166007996], [0.18983790278434753, 0.15936079621315002, 0.4752041697502136, 0.008670181035995483], [0.07695235311985016, 0.12814626097679138, 0.24936996400356293, 0.28386378288269043], [0.07695235311985016, 0.12814626097679138, 0.24936996400356293, 0.28386378288269043]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_c6f90a81774a5479d094ff5095d1f1a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_ce4cc696132755dce4404c3daba0b98d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_016df1ebed499528eeefbc832767faf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_7704e705c7887c52cbf55d5ec59d1ab4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_740f16aec91797da3e051acb5494e125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dfb35d68f1a2ad8643b9afa937c4c01b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_dfb35d68f1a2ad8643b9afa937c4c01b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_bc88f57628a852d4c30cbf50b6212390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_3c1b41c6e0636a35cff609143b296146(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ae87674db7133e8bb5031c55f9b7f363(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10831242054700851, 0.13026173412799835, 0.21035446226596832, 0.1508524864912033, 0.02271188236773014, 0.12964244186878204, 0.01880607381463051, 0.04555300250649452, 0.004955768585205078, 0.11307639628648758, 0.16117045283317566, 0.1519143134355545, 0.19969022274017334, 0.1847439408302307, 0.046241242438554764, 0.1009218618273735, 0.23774173855781555, 0.16711412370204926, 0.19791994988918304, 0.20945854485034943, 0.18047094345092773, 0.05879276990890503, 0.105992890894413, 0.13922762870788574], dtype='float32').reshape([24]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_780cb907b66b2aa6d9553c0576dd80e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_62d5ec799740261908275d3f1decd616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([1565, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_62d5ec799740261908275d3f1decd616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([1565, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_92ed0a70317218f94262bddc014a8c86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_20dcc662b00fccd4f1183695b2631235(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.15230226516723633, 0.2131345123052597, 0.17228750884532928, 0.20414535701274872], dtype='float32').reshape([4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_016df1ebed499528eeefbc832767faf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_b6daf6eb8fac6ee93a8923bfdfa5333d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07388460636138916, 0.34592828154563904, 0.13610978424549103, 0.030362337827682495], [0.14449560642242432, 0.15574820339679718, 0.3628971576690674, 0.059735897928476334], [0.22425659000873566, 0.20642846822738647, 0.19675441086292267, 0.21475961804389954], [0.04178471863269806, 0.07089582085609436, 0.4266142249107361, 0.2633800506591797], [0.04178471863269806, 0.07089582085609436, 0.4266142249107361, 0.2633800506591797], [0.22425659000873566, 0.20642846822738647, 0.19675441086292267, 0.21475961804389954]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_016df1ebed499528eeefbc832767faf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_3660fdc542062277b43ec48de01ced78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2612953782081604, 0.28655803203582764, 0.29477638006210327, 0.2740551829338074], [0.07684624195098877, 0.028802603483200073, 0.05022908002138138, 0.10331213474273682], [0.09165571630001068, 0.16482195258140564, 0.15293759107589722, 0.1458241045475006], [0.23540610074996948, 0.005343496799468994, 0.14588503539562225, 0.05192685127258301], [0.2612953782081604, 0.28655803203582764, 0.29477638006210327, 0.2740551829338074]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_016df1ebed499528eeefbc832767faf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_2b93c152c74ebf31c5b0ed81d2328c3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_92ed0a70317218f94262bddc014a8c86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_016df1ebed499528eeefbc832767faf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_aed4e505bf849e911d5b5f1ff3d37f90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.38793617486953735, 0.0617041140794754, 0.16063280403614044, 0.06832501292228699], [0.14784809947013855, 0.2535051703453064, 0.0055626630783081055, 0.2679281234741211], [0.2185918092727661, 0.24922549724578857, 0.26665961742401123, 0.18001899123191833], [0.19315892457962036, 0.292204886674881, 0.23457635939121246, 0.3767058849334717]], dtype='float32').reshape([4, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class PrimitiveOp_699eb4e5eb57d0f1079fefbd78a0e769(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.sum(input_0, input_1, None, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e78a98d1e124a8fe1d0bbfc77ddb1d9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699eb4e5eb57d0f1079fefbd78a0e769
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3c1b41c6e0636a35cff609143b296146(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_016df1ebed499528eeefbc832767faf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_0eeb670329fe78b83659bdf12228b92b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_e6c1711dd07d22864fb1083b68022543(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_80f2979e6d4f78d92ee704e4df8db784(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_79c535b8292a8c8dda0f68cc0ac012d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bfe0873ab5f4886f95f912d5d854578e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([2034, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_bfe0873ab5f4886f95f912d5d854578e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([2034, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_1320774a6682e61306d9b1c5a19576ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_699eb4e5eb57d0f1079fefbd78a0e769
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_016df1ebed499528eeefbc832767faf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_1faf3fe0d2f87a4c377d323d3dfdc0e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17936022579669952, 0.029575109481811523, 0.3104240298271179, 0.019696682691574097], [0.17936022579669952, 0.029575109481811523, 0.3104240298271179, 0.019696682691574097], [0.2813943028450012, 0.1955566108226776, 0.018820174038410187, 0.11219587922096252], [0.0066500455141067505, 0.046088606119155884, 0.012532830238342285, 0.1825047880411148], [0.35801857709884644, 0.08647437393665314, 0.11524057388305664, 0.3932095468044281], [0.08881855010986328, 0.07655680179595947, 0.21666808426380157, 0.44117167592048645], [0.3748916983604431, 0.12003204226493835, 0.022046446800231934, 0.014015290886163712]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_aa0712a4d511f25b8fc96468e4e0451d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6a27d75ded3831354e76890ae30c563f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([4667, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_6a27d75ded3831354e76890ae30c563f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([4667, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_89ea522e396cb6ab2313375dc05cbf87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([4921], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_11ac6e17bfe239ce0122e399c0610be3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([1233], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_accfce56e0a58792d9f25cf51d8b1b68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_285ec15c38edf0a15f5dd824273b5b6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b756793853fd9bfb92b48028f6f12611(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([1052, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_b756793853fd9bfb92b48028f6f12611(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([1052, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_016df1ebed499528eeefbc832767faf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_ef077d261038c9a2103322cee5993d81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06581726670265198, 0.012930139899253845, 0.31881576776504517, 0.04646962881088257], [0.40379393100738525, 0.012253254652023315, 0.35942530632019043, 0.08588998019695282], [0.40379393100738525, 0.012253254652023315, 0.35942530632019043, 0.08588998019695282], [0.08568122982978821, 0.09530001878738403, 0.22064247727394104, 0.42547687888145447], [0.3612695634365082, 0.2549719512462616, 0.25649890303611755, 0.2786141335964203], [0.14194762706756592, 0.3295552432537079, 0.39786407351493835, 0.16491538286209106]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_a5caa0fe1b26326f54014d36181488c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8f830bfefcafe28703e4546e613e0dcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2c3e97c4caece6a2702a467d83b6a73b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d1f6a68e19bba5ad1d20f7db59a75dda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_aa2130ba2395be194d6c990598876cf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([2378, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_aa2130ba2395be194d6c990598876cf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([2378, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_5f837eb8fd8437ff86073c61ad52745d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a95adc40aa9a8702148d6a0ad515cd26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([3105, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_a95adc40aa9a8702148d6a0ad515cd26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([3105, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_06c5b033d4616f4bbad04c57e385ea57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b059f372d0b8ab52f907d1d6701677d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_b059f372d0b8ab52f907d1d6701677d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_89aee509f2db458b2367faade6ccf08e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_1a19554fb3bec3ecba1c564d8cf5ae71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cdc6a9409f5d1986f148ac1c4c078255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ce4cc696132755dce4404c3daba0b98d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_67ebad26fe3c79278ac4b528260cd0f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0033390524331480265, 0.21986597776412964, 0.10636845976114273, 0.2046603560447693, 0.10598552227020264, 0.1594502478837967, 0.0007593439077027142, 0.10296109318733215, 0.20096901059150696, 0.08761290460824966, 0.2456406205892563, 0.22014686465263367, 0.21511882543563843, 0.1623859703540802, 0.24103717505931854, 0.01302557997405529, 0.13736790418624878, 0.14086657762527466, 0.15425017476081848, 0.24744603037834167], dtype='float32').reshape([20]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_ccb8778a171a6a47454fe29e5de17605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([17421], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_1a19554fb3bec3ecba1c564d8cf5ae71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0dc1a3ce4eccfdd7b0e3e5fb3f7a1232(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_b7b84fb5ca797cd9dcdec32576996659(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_58b591c05561982bde82dc910a5a5fba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1dae00df523e697d5e860b3eff2c6a8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([2087, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_1dae00df523e697d5e860b3eff2c6a8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([2087, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_714d43297fc2e17e4d87a2b64bc1c525(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f2f46b0f359fc9937ab4b0db9932f585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_016df1ebed499528eeefbc832767faf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_96a751fb9dc6d549cebad25d1709e026(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08812366425991058, 0.028453081846237183, 0.10375607013702393, 0.09819439053535461], [0.15779852867126465, 0.016703754663467407, 0.062226682901382446, 0.12866711616516113], [0.14701034128665924, 0.0776543989777565, 0.2557470202445984, 0.01866358518600464], [0.14701034128665924, 0.0776543989777565, 0.2557470202445984, 0.01866358518600464], [0.2722809612751007, 0.014804542064666748, 0.3420112133026123, 0.3253910541534424]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_8f830bfefcafe28703e4546e613e0dcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4f1f1eed2f983e766f06b744f4a9d22f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_211305b099701314838e05e8dad37935(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_16524d18aa7445818d5ac81ad849be02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_624c0e1b99bd7803371455cd510c6db4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ced0d018def9586f4026d70772d35867
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4a794a32a042db391f4ef31d86ad2a8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([4271, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_4a794a32a042db391f4ef31d86ad2a8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([4271, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_016df1ebed499528eeefbc832767faf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a371de56f980181327a77e57389669bb
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_8bb572ae6eb3af0c58204a112c419a4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17719396948814392, 0.0036325454711914062, 0.1302030086517334, 0.29399293661117554], [0.25933337211608887, 0.26464760303497314, 0.024477168917655945, 0.1406756043434143], [0.08580110222101212, 0.012520700693130493, 0.13238537311553955, 0.24548061192035675], [0.17719396948814392, 0.0036325454711914062, 0.1302030086517334, 0.29399293661117554], [0.18552877008914948, 0.008648484945297241, 0.4290626347064972, 0.21437333524227142], [0.026892393827438354, 0.03062039241194725, 0.22371087968349457, 0.15546290576457977], [0.18552877008914948, 0.008648484945297241, 0.4290626347064972, 0.21437333524227142]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_6156579b859287509a0e27de1cb8d224(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e67340cf34eda9fba45364ddcf6ac32
    def get_inputs(self):
        return [
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_cdc6a9409f5d1986f148ac1c4c078255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94de098bf59f51bc5f318e663b40eef7
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()