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


class TestPrimitiveOp_9ce500d412a700d51c7ac4cd6916d886(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([4421], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_5d94bfdad053545e4498f935cd0d43f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c34fcd8b1bb8a1a709a2d8de518d1e2d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.009965404868125916, 0.06563380360603333]], [[0.05196989327669144, 0.049899157136678696]], [[0.11760067194700241, 0.058422185480594635]], [[0.0005940748378634453, 0.0010154024930670857]], [[0.004336654674261808, 0.16057434678077698]], [[0.09119416028261185, 0.19754429161548615]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c6c7cc291467279dca782087455d46d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c34fcd8b1bb8a1a709a2d8de518d1e2d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.008592716418206692, 0.021761853247880936]], [[0.002021761378273368, 0.06571615487337112]], [[0.05858742818236351, 0.001349923200905323]], [[0.0023106774315238, 0.08131388574838638]], [[0.11052302271127701, 0.21619395911693573]], [[0.00010153105540666729, 0.024723242968320847]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class TestPrimitiveOp_e9ccfa4479625eb0c57ddbbb3679d812(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.to_tensor([0.26197555661201477, 0.04112504422664642, 0.1651178002357483, 0.22243663668632507, 0.1741013526916504, 0.05572321638464928, 0.19509509205818176, 0.07952500879764557, 0.0693402886390686, 0.048220109194517136, 0.14490318298339844, 0.09243150800466537, 0.16872058808803558, 0.006763482932001352, 0.12934298813343048, 0.04820796474814415], dtype='float32').reshape([16]),
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


class TestPrimitiveOp_184edcc1995f8486c56ccdaced1d1e7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_184edcc1995f8486c56ccdaced1d1e7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_e95ac35f31b132b9657553d2f18834c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.23606227338314056, 0.1306420862674713, 0.056651972234249115, 0.15376406908035278], [0.0829421803355217, 0.21722771227359772, 0.06370577216148376, 0.11743849515914917], [0.32634949684143066, 0.11071017384529114, 0.04787001013755798, 0.39140892028808594], [0.34305647015571594, 0.3147853910923004, 0.21061715483665466, 0.21905085444450378], [0.09009411931037903, 0.12927646934986115, 0.13630236685276031, 0.38561052083969116]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_d8b5d7d4bb0eaccfed40e23d837ceef2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10176767408847809, 0.027969181537628174, 0.13815650343894958, 0.07590332627296448], [0.3440432548522949, 0.017140405252575874, 0.02755213901400566, 0.09486806392669678], [0.1388358175754547, 0.20331719517707825, 0.34952691197395325, 0.289692759513855], [0.3440432548522949, 0.017140405252575874, 0.02755213901400566, 0.09486806392669678], [0.1388358175754547, 0.20331719517707825, 0.34952691197395325, 0.289692759513855]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_8b77b920c588f5956207b44f8845e1df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_8b77b920c588f5956207b44f8845e1df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_204e5ff211bd2c8b04835542ed74e797(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1511559784412384, 0.16333091259002686, 0.000880807638168335, 0.04097789525985718], [0.06759227812290192, 0.11715888977050781, 0.005081087350845337, 0.07123449444770813], [0.0758233442902565, 0.31037062406539917, 0.3314056992530823, 0.09279140830039978], [0.06759227812290192, 0.11715888977050781, 0.005081087350845337, 0.07123449444770813], [0.0758233442902565, 0.31037062406539917, 0.3314056992530823, 0.09279140830039978], [0.06419932842254639, 0.023158907890319824, 0.032714784145355225, 0.24787023663520813], [0.06419932842254639, 0.023158907890319824, 0.032714784145355225, 0.24787023663520813]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_286b3c45731e639f73f633d124cc05db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_286b3c45731e639f73f633d124cc05db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_b2fa64afdb37ddf066aeaae1e0edef97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20068788528442383, 0.1586471050977707, 0.16377636790275574, 0.06369337439537048, 0.10820113867521286, 0.1860961765050888, 0.006097310222685337, 0.17595161497592926, 0.22940798103809357, 0.10722284764051437, 0.07986682653427124, 0.12861722707748413, 0.15673448145389557, 0.16781432926654816, 0.07254330813884735, 0.21343214809894562, 0.057397689670324326, 0.09165479242801666, 0.26153039932250977, 0.024889584630727768, 0.0592675544321537, 0.19692778587341309, 0.04139372333884239, 0.09711730480194092], dtype='float32').reshape([24]),
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


class TestPrimitiveOp_2f1d291ad9e7e8872cfe8db660c9dc84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_2f1d291ad9e7e8872cfe8db660c9dc84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_a13cc8cdc8a50642ee1221d8ba96fa71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2411714494228363, 0.21830829977989197, 0.15804500877857208, 0.15290726721286774], dtype='float32').reshape([4]),
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


class TestPrimitiveOp_593b3563ef01e5b1be2e91027c6dee48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03431218862533569, 0.026114583015441895, 0.3125544786453247, 0.05361819267272949], [0.2707841992378235, 0.19268369674682617, 0.006156504154205322, 0.09308546781539917], [0.10481724143028259, 0.22131045162677765, 0.2644655704498291, 0.28144195675849915], [0.18018700182437897, 0.11127512902021408, 0.4410821199417114, 0.0970020592212677], [0.18018700182437897, 0.11127512902021408, 0.4410821199417114, 0.0970020592212677], [0.10481724143028259, 0.22131045162677765, 0.2644655704498291, 0.28144195675849915]], dtype='float32').reshape([6, 4]),
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


class TestPrimitiveOp_cfba868afcb67455e272899b5157ef5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.202517569065094, 0.2943516671657562, 0.30916333198547363, 0.01913774013519287], [0.2773173451423645, 0.03562504053115845, 0.11010816693305969, 0.3304157257080078], [0.3562518060207367, 0.21646155416965485, 0.03921368718147278, 0.10079234093427658], [0.05663609504699707, 0.09692704677581787, 0.04421214759349823, 0.12598776817321777], [0.202517569065094, 0.2943516671657562, 0.30916333198547363, 0.01913774013519287]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_316b6e29fde98c5e57e596a9cf7470a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.36131948232650757, 0.3353999853134155, 0.02784162014722824, 0.07963600009679794], [0.41048651933670044, 0.3893478214740753, 0.2992238700389862, 0.15474331378936768], [0.08667996525764465, 0.19611337780952454, 0.06163033843040466, 0.023111797869205475], [0.1219969391822815, 0.08688667416572571, 0.3911390006542206, 0.2473030537366867]], dtype='float32').reshape([4, 4]),
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


class TestPrimitiveOp_6ea7d8e81173f930622061c6e8210fd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_6ea7d8e81173f930622061c6e8210fd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_88360f62207f9f66f8650c8d53a639fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3118785619735718, 0.06509244441986084, 0.29392749071121216, 0.3088516592979431], [0.3118785619735718, 0.06509244441986084, 0.29392749071121216, 0.3088516592979431], [0.16390565037727356, 0.3733579218387604, 0.2686590552330017, 0.21544930338859558], [0.2529969811439514, 0.006404057145118713, 0.19852834939956665, 0.1327028125524521], [0.0964447557926178, 0.10411974787712097, 0.06578028202056885, 0.07751882076263428], [0.4481545686721802, 0.07796558737754822, 0.0771525502204895, 0.06475737690925598], [0.20786544680595398, 0.2283610701560974, 0.08375959098339081, 0.08007444441318512]], dtype='float32').reshape([7, 4]),
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


class TestPrimitiveOp_053d766de61c6d4d5ace58e73f6e6472(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_053d766de61c6d4d5ace58e73f6e6472(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_96a019f40dc97191c64094c1b154aa1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([4901], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_8aef37eaf7a1e1c543dde4d505a8e0af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([1247], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_ddd9c3f493c5d4e8fe85e4922e3c0dfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_ddd9c3f493c5d4e8fe85e4922e3c0dfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_80fc8800def807e8a5f8b0da4bb060c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07373276352882385, 0.1829991638660431, 0.21339315176010132, 0.003011345863342285], [0.028881244361400604, 0.03423634171485901, 0.43788984417915344, 0.17044034600257874], [0.028881244361400604, 0.03423634171485901, 0.43788984417915344, 0.17044034600257874], [0.13281306624412537, 0.10249564051628113, 0.12424007803201675, 0.47744619846343994], [0.18827302753925323, 0.06766808032989502, 0.022206313908100128, 0.06729784607887268], [0.1352621465921402, 0.3263186514377594, 0.1006387323141098, 0.21034857630729675]], dtype='float32').reshape([6, 4]),
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


class TestPrimitiveOp_fa6d5651a79aa13735ee78a6a9beae97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_fa6d5651a79aa13735ee78a6a9beae97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_a7d5cdb644348b352d47decafd980b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_a7d5cdb644348b352d47decafd980b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_0a1202be18ddce7c649293c8601c2e21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_0a1202be18ddce7c649293c8601c2e21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_eecc46079fd58e12828ba96d42a46735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.to_tensor([0.19897453486919403, 0.008862131275236607, 0.07665710896253586, 0.14328773319721222, 0.1754627525806427, 0.14245685935020447, 0.14736685156822205, 0.08872371166944504, 0.14866097271442413, 0.012519782409071922, 0.15323063731193542, 0.17460323870182037, 0.052720796316862106, 0.10413170605897903, 0.1659744530916214, 0.27285057306289673, 0.07536588609218597, 0.18901333212852478, 0.24205811321735382, 0.16555249691009521], dtype='float32').reshape([20]),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_cc98198accb3df88274babb2a9dce25c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e17f982ae4f0e1eb697585fd5763df12
    def get_inputs(self):
        return [
            paddle.uniform([17350], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_c1a3099ebc5e82bffbddf528a500a53c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_c1a3099ebc5e82bffbddf528a500a53c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_76a862a45435b10ff6f7708b7fceeb4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3232983648777008, 0.26429325342178345, 0.032824814319610596, 0.06961558759212494], [0.10855147242546082, 0.23897580802440643, 0.12071990966796875, 0.02904871106147766], [0.039555326104164124, 0.10271425545215607, 0.11992594599723816, 0.08371031284332275], [0.039555326104164124, 0.10271425545215607, 0.11992594599723816, 0.08371031284332275], [0.25436314940452576, 0.24295449256896973, 0.14639168977737427, 0.03259342908859253]], dtype='float32').reshape([5, 4]),
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


class TestPrimitiveOp_47347017d38d0b10f39b3fd9f2fbaf40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64'),
        ]


class TestPrimitiveOp_47347017d38d0b10f39b3fd9f2fbaf40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_8d53d38b2a1953a347f30c65fdbada7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c9c0b69a78f41d012122e0f6a1c62a9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19802163541316986, 0.16961577534675598, 0.15227729082107544, 0.025044582784175873], [0.2353224754333496, 0.05387616157531738, 0.14789704978466034, 0.0630868673324585], [0.001409083604812622, 0.05223914980888367, 0.08461788296699524, 0.06864841282367706], [0.19802163541316986, 0.16961577534675598, 0.15227729082107544, 0.025044582784175873], [0.03428974747657776, 0.11960519850254059, 0.20580589771270752, 0.07139194011688232], [0.0854860246181488, 0.07616549730300903, 0.29124677181243896, 0.13023613393306732], [0.03428974747657776, 0.11960519850254059, 0.20580589771270752, 0.07139194011688232]], dtype='float32').reshape([7, 4]),
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