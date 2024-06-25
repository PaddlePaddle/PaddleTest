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



class PrimitiveOp_365a3e4c51a6c14309877f2753446bc7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa3b35b756c3070ac30648592d448e99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365a3e4c51a6c14309877f2753446bc7
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ad9f7e596842429edbc9af80e8aa8af9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([4431], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_e9db95ae85711a12240ad682dc339ade(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_13e401748933a8933b193cc99d2a1c5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9db95ae85711a12240ad682dc339ade
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_3249489623a8aac393d12cdbd1d8a7f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc5013c807082eddc43ab057cacb503c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_c34fcd8b1bb8a1a709a2d8de518d1e2d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c13b9c8ba48a8d3c2ff4ca02a3007faf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c34fcd8b1bb8a1a709a2d8de518d1e2d
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c13b9c8ba48a8d3c2ff4ca02a3007faf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c34fcd8b1bb8a1a709a2d8de518d1e2d
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7979b4ba633a65e4494bdaf135bb7b04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c34fcd8b1bb8a1a709a2d8de518d1e2d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.2970739741576836e-05, 0.026655826717615128]], [[0.051871638745069504, 0.09213211387395859]], [[0.03389995917677879, 0.17288225889205933]], [[0.01788737066090107, 0.12511767446994781]], [[0.007303297054022551, 0.03130850940942764]], [[0.02491130121052265, 0.05975374951958656]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_e92fcc5741e21d2fe73d51866d730a1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c34fcd8b1bb8a1a709a2d8de518d1e2d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.002407881896942854, 0.0007964816177263856]], [[0.04602523893117905, 0.11046715080738068]], [[0.006478022318333387, 0.0500929057598114]], [[0.0038370443508028984, 0.01831933669745922]], [[0.004518288187682629, 0.031191511079669]], [[0.0256083682179451, 0.007284558843821287]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_579c8a9e598357f8e3f4abe6a7954acc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365a3e4c51a6c14309877f2753446bc7
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_77a7630f40504c8dbc8a728d23109cea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14544907212257385, 0.16143012046813965, 0.12652279436588287, 0.01769750565290451, 0.15413163602352142, 0.07005314528942108, 0.14634035527706146, 0.2093019187450409, 0.06052927300333977, 0.043568722903728485, 0.26218268275260925, 0.04366888105869293, 0.14885859191417694, 0.19055111706256866, 0.22633613646030426, 0.059489961713552475], dtype='float32').reshape([16]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_7c6d4159706fe73a179647de059fbc05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6b7949dd7d365e5ee68847fd4c10a30d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_5f4844adfe91a4079b95f8521b70a71c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_56cebce22c6ffdac6af32be45d53560a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1533e7bf25b1fb7e55190373cfde2980(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56cebce22c6ffdac6af32be45d53560a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a2fe4f126bf14c6729ce846f8d614d57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_a2fe4f126bf14c6729ce846f8d614d57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([1762, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_c5f9930f25a9104451328734567e80fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_3249489623a8aac393d12cdbd1d8a7f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_050596b0011fa6ac700221edf495fe68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06151318550109863, 0.11936834454536438, 0.07800794392824173, 0.267497181892395], [0.11394819617271423, 0.005255617201328278, 0.09603127837181091, 0.057251498103141785], [0.2812342643737793, 0.2772517204284668, 0.2135857790708542, 0.19352394342422485], [0.05876466631889343, 0.2864294648170471, 0.11026737093925476, 0.3292335867881775], [0.22799256443977356, 0.13594946265220642, 0.23514008522033691, 0.044174402952194214]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_37f7936a06e61b4f4790f00b165b84c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365a3e4c51a6c14309877f2753446bc7
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3249489623a8aac393d12cdbd1d8a7f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_1b6b251815710250fef6817bb19f10af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.26748061180114746, 0.13443715870380402, 0.014436036348342896, 0.42332661151885986], [0.11801926791667938, 0.1759794056415558, 0.27236711978912354, 0.30360615253448486], [0.06403589248657227, 0.21310356259346008, 0.02465561032295227, 0.3234819173812866], [0.11801926791667938, 0.1759794056415558, 0.27236711978912354, 0.30360615253448486], [0.06403589248657227, 0.21310356259346008, 0.02465561032295227, 0.3234819173812866]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_1c12e4ad810fb0a978370f9d709a942d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9db95ae85711a12240ad682dc339ade
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_fc5d7961e2b253a8a5d6edd2e4050056(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56cebce22c6ffdac6af32be45d53560a
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_af29eb7595898c1e7336e03effc93811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_af29eb7595898c1e7336e03effc93811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([5522, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_3249489623a8aac393d12cdbd1d8a7f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_bc6a9f9f7788960a5da23f9afb675576(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06942878663539886, 0.39978697896003723, 0.04090121388435364, 0.11602064967155457], [0.0561792254447937, 0.21276485919952393, 0.11734984070062637, 0.08785513043403625], [0.1648823320865631, 0.051890671253204346, 0.014645934104919434, 0.06937026977539062], [0.0561792254447937, 0.21276485919952393, 0.11734984070062637, 0.08785513043403625], [0.1648823320865631, 0.051890671253204346, 0.014645934104919434, 0.06937026977539062], [0.20225289463996887, 0.25074583292007446, 0.2275557518005371, 0.05374142527580261], [0.20225289463996887, 0.25074583292007446, 0.2275557518005371, 0.05374142527580261]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_129222b48aa2df20e45e95d18a3b1861(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_4400ef64f958b63ab4078848957ccf20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365a3e4c51a6c14309877f2753446bc7
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3249489623a8aac393d12cdbd1d8a7f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_fe3a2732daf3b0a7c526d6fae938f7a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_1533e7bf25b1fb7e55190373cfde2980(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56cebce22c6ffdac6af32be45d53560a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b8df325fb012608a7eaada8be75e1302(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b8df325fb012608a7eaada8be75e1302(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([1760, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_c5f9930f25a9104451328734567e80fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_d35a3daf34fda899fdcabdb651a69dfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365a3e4c51a6c14309877f2753446bc7
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_93d1cf896ed4eb103d6be9b7926792f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09630738943815231, 0.06467706710100174, 0.12173382937908173, 0.11143457889556885, 0.23838429152965546, 0.1329624205827713, 0.10066625475883484, 0.057149045169353485, 0.12025514245033264, 0.017536723986268044, 0.12403053045272827, 0.06292761862277985, 0.18012507259845734, 0.04825600981712341, 0.2304755002260208, 0.1837431937456131, 0.2350190281867981, 0.2532053589820862, 0.16433703899383545, 0.11874276399612427, 0.0579327754676342, 0.13508449494838715, 0.124302439391613, 0.019513379782438278], dtype='float32').reshape([24]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_988a2dc3b79b5645081df91ca314913e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56cebce22c6ffdac6af32be45d53560a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_809ad418b2954bcdc9c4ddd0b4335a25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_809ad418b2954bcdc9c4ddd0b4335a25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([1522, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_917d3008955226a96d2896d03fb983e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365a3e4c51a6c14309877f2753446bc7
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2ea9916428deeebc134f45723ac1f5c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.to_tensor([0.006086781620979309, 0.21291619539260864, 0.04403847083449364, 0.16496078670024872], dtype='float32').reshape([4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_3249489623a8aac393d12cdbd1d8a7f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_3ebc7e617e4ffc45219ac91c7daf9350(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10981824994087219, 0.1818036437034607, 0.005384296178817749, 0.17434953153133392], [0.12394212931394577, 0.34773626923561096, 0.10430505871772766, 0.23078596591949463], [0.2615264356136322, 0.13417835533618927, 0.25303182005882263, 0.0923982560634613], [0.4147195816040039, 0.29263797402381897, 0.10034643113613129, 0.3984808325767517], [0.4147195816040039, 0.29263797402381897, 0.10034643113613129, 0.3984808325767517], [0.2615264356136322, 0.13417835533618927, 0.25303182005882263, 0.0923982560634613]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_3249489623a8aac393d12cdbd1d8a7f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_a30496d43fa10ae62977e07554b9bad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1289084106683731, 0.24571344256401062, 0.264828622341156, 0.1877102255821228], [0.04371963441371918, 0.03634536266326904, 0.05285709723830223, 0.056930214166641235], [0.34693601727485657, 0.18697671592235565, 0.08248449862003326, 0.23773470520973206], [0.18576744198799133, 0.06885099411010742, 0.14518606662750244, 0.03412967920303345], [0.1289084106683731, 0.24571344256401062, 0.264828622341156, 0.1877102255821228]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_3249489623a8aac393d12cdbd1d8a7f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_16554a2fa008fbe948328c98e836e47c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_917d3008955226a96d2896d03fb983e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365a3e4c51a6c14309877f2753446bc7
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3249489623a8aac393d12cdbd1d8a7f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_40b2d2b95c34722dba30e250b24079bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.030229568481445312, 0.15371039509773254, 0.21909597516059875, 0.0034876130521297455], [0.09754230082035065, 0.1374218612909317, 0.24284592270851135, 0.06160402297973633], [0.06366714835166931, 0.1073785126209259, 0.3071011006832123, 0.03405541181564331], [0.09629049897193909, 0.00018036365509033203, 0.1322488784790039, 0.40136411786079407]], dtype='float32').reshape([4, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class PrimitiveOp_2a2eee077d2de26ddd0f3e0065176ca8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.sum(input_0, input_1, None, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8fd6e0c68c37a53818eb0a43a64b1a9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a2eee077d2de26ddd0f3e0065176ca8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d35a3daf34fda899fdcabdb651a69dfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365a3e4c51a6c14309877f2753446bc7
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_3249489623a8aac393d12cdbd1d8a7f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b8c80882e24069db8dd62bf605520bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_fc5d8f66db76cfedc5241dccea51705a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_acbb00721ea026f1e7af501c1a05d1ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b118ea294938c2aee4d62580a52ff076(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56cebce22c6ffdac6af32be45d53560a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f90e02e968efbe3071a945283a63043b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_f90e02e968efbe3071a945283a63043b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([2074, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_0a7f5318fe91af455eab18e123b36009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a2eee077d2de26ddd0f3e0065176ca8
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_3249489623a8aac393d12cdbd1d8a7f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_409d353e2492c0ad285b2511788100b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17910704016685486, 0.3516285717487335, 0.11908003687858582, 0.24569259583950043], [0.17910704016685486, 0.3516285717487335, 0.11908003687858582, 0.24569259583950043], [0.02250000834465027, 0.35152024030685425, 0.15124770998954773, 0.09407028555870056], [0.16901521384716034, 0.23902814090251923, 0.1676264852285385, 0.28854885697364807], [0.1521880328655243, 0.27202269434928894, 0.07244840264320374, 0.10833775997161865], [0.13918226957321167, 0.06347241997718811, 0.27488210797309875, 0.3045266270637512], [0.02585412561893463, 0.12091132253408432, 0.24972589313983917, 0.03646087646484375]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_c4b9a408c3ab296c9b3bd2512efa8250(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56cebce22c6ffdac6af32be45d53560a
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5f0844c226172d2839e2601ce5515514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_5f0844c226172d2839e2601ce5515514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([4734, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_5939bfdca582930649c4c01512cead0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([4861], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_9c2a513658524028240b339f6ecc3a3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([1253], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_bfbe9dbb8cc8a6326602cae3f14a7830(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9db95ae85711a12240ad682dc339ade
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_481f9bc21a9b98a5516504fc34e37b05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56cebce22c6ffdac6af32be45d53560a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f4e19ce8b437da9459e8bfc5dc2d8f1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_f4e19ce8b437da9459e8bfc5dc2d8f1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_3249489623a8aac393d12cdbd1d8a7f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_e2b91b48283bd921d0ef5074680d982a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17867141962051392, 0.038635507225990295, 0.10939210653305054, 0.22395551204681396], [0.01621919870376587, 0.3006782531738281, 0.033404335379600525, 0.06576906144618988], [0.01621919870376587, 0.3006782531738281, 0.033404335379600525, 0.06576906144618988], [0.2828966975212097, 0.11116990447044373, 0.3995091915130615, 0.10126431286334991], [0.2237587720155716, 0.027776330709457397, 0.017031311988830566, 0.014228790998458862], [0.1819733828306198, 0.3137103319168091, 0.27327537536621094, 0.2752225995063782]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_a87db7067ca65645d0f3e5e7ad271c43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56cebce22c6ffdac6af32be45d53560a
    def get_inputs(self):
        return [
            paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4dd0392572eca1c6ce361d87f636f608(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365a3e4c51a6c14309877f2753446bc7
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8a8327ac00d554c050a0531f17a9b5ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56cebce22c6ffdac6af32be45d53560a
    def get_inputs(self):
        return [
            paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_91411a379c411e13ddef84ca7eb3eb48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56cebce22c6ffdac6af32be45d53560a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_547d98ab7b10fc0a418ccce563be63e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_547d98ab7b10fc0a418ccce563be63e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([2332, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_1bf1fcb9680b39f68ff7f7f8947761c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56cebce22c6ffdac6af32be45d53560a
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d241c1548b807c9ae6d74619dc65a36b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_d241c1548b807c9ae6d74619dc65a36b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([3051, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_bc1277a8ebabee2772e9b7cfd2bc1512(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56cebce22c6ffdac6af32be45d53560a
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_93e3c7761ca10fa6b2cc7bb00742c164(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_93e3c7761ca10fa6b2cc7bb00742c164(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([3870, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_d0aaf30a14ee26e91c33634f19568beb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_d89e93ce285045a9e4dbf75f0488bfd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365a3e4c51a6c14309877f2753446bc7
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c92310610d7bf137ad3f46483eb6178e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365a3e4c51a6c14309877f2753446bc7
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4400ef64f958b63ab4078848957ccf20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365a3e4c51a6c14309877f2753446bc7
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d6212b1f91f8f6cd54d72ee59b6efa07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11011484265327454, 0.09124531596899033, 0.09387259930372238, 0.2433108240365982, 0.09141333401203156, 0.03867831453680992, 0.17261779308319092, 0.005376485642045736, 0.024394644424319267, 0.12686291337013245, 0.14892083406448364, 0.10076786577701569, 0.15849126875400543, 0.07344936579465866, 0.14530909061431885, 0.11683167517185211, 0.08212780952453613, 0.15044739842414856, 0.23034560680389404, 0.18885627388954163], dtype='float32').reshape([20]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_7dd97ce89622a15f4b77ae5c0907453a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([17538], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_d89e93ce285045a9e4dbf75f0488bfd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365a3e4c51a6c14309877f2753446bc7
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6cf486b1d9ad40c9501aab34517c48bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_d56f1d2171e15640436a9f8611bd2846(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_86c124084df99972b2c80adfa98391e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56cebce22c6ffdac6af32be45d53560a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ea08e9dbdf3c55cf630a463dc410f878(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_ea08e9dbdf3c55cf630a463dc410f878(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([2111, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_68566ef1368252e09e3b9952b15176e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365a3e4c51a6c14309877f2753446bc7
    def get_inputs(self):
        return [
            paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1283ab9bcd8099b0d06c73c868fe78e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_3249489623a8aac393d12cdbd1d8a7f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_49b6459cfa255fbe4378c6efc3286801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.13473466038703918, 0.12285895645618439, 0.10564139485359192, 0.24786412715911865], [0.34976115822792053, 0.31551921367645264, 0.26948869228363037, 0.12118439376354218], [0.2146769016981125, 0.40058374404907227, 0.1350015103816986, 0.13491290807724], [0.2146769016981125, 0.40058374404907227, 0.1350015103816986, 0.13491290807724], [0.0060720741748809814, 0.19148385524749756, 0.026892781257629395, 0.30257436633110046]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_4dd0392572eca1c6ce361d87f636f608(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365a3e4c51a6c14309877f2753446bc7
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2c5afcd95cabd692667f90321a770b34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_e60fe145dd7bb041a23e60e9cd18b82e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_b9322f64513ec78e9f5bb8ffb23537b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_dee068a61b4ea5f0245ebf547acb9d09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56cebce22c6ffdac6af32be45d53560a
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_904ce06ed4e7e28851cef880ef43f28a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_904ce06ed4e7e28851cef880ef43f28a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([4141, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_3249489623a8aac393d12cdbd1d8a7f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_a633e7a1de764c78feff5e3a8a42e0a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14209389686584473, 0.06900376081466675, 0.2527446746826172, 0.19547338783740997], [0.03495118021965027, 0.08211711049079895, 0.1606784164905548, 0.3424886167049408], [0.0032216906547546387, 0.25409072637557983, 0.05634039640426636, 0.21551376581192017], [0.14209389686584473, 0.06900376081466675, 0.2527446746826172, 0.19547338783740997], [0.48535096645355225, 0.050888895988464355, 0.054100051522254944, 0.11393816769123077], [0.10485643148422241, 0.06027994304895401, 0.3643565773963928, 0.058671534061431885], [0.48535096645355225, 0.050888895988464355, 0.054100051522254944, 0.11393816769123077]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_253a7c345c6c225f258f49ce338d7758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_c92310610d7bf137ad3f46483eb6178e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365a3e4c51a6c14309877f2753446bc7
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()