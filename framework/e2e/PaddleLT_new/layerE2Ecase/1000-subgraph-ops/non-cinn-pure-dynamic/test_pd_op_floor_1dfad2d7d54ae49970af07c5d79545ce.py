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



class PrimitiveOp_463abca34f485cc923eabecd39128569(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fb71ab720bffeef2f2ecbd45256305dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.7638370990753174]]], [[[1.474405288696289]]], [[[1.421640396118164]]], [[[1.2275431156158447]]], [[[1.5212799310684204]]], [[[1.3487391471862793]]], [[[1.161902666091919]]], [[[1.234742283821106]]], [[[1.5890226364135742]]], [[[1.3468501567840576]]], [[[1.4928845167160034]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_651a19819dedc4b7d3473fa9edacca00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_651a19819dedc4b7d3473fa9edacca00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d4d3cbab577d074dccd5e86586288195(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a55bcf69d8c049413fe901859a93c2b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([1745, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ee22cf13dae17e8ed1cb1599237b578(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.420096516609192]]], [[[1.2214287519454956]]], [[[0.9462398886680603]]], [[[1.6496424674987793]]], [[[1.538584589958191]]], [[[1.3461487293243408]]], [[[1.8142749071121216]]], [[[1.3717825412750244]]], [[[1.4882677793502808]]], [[[0.9728361368179321]]], [[[1.2029166221618652]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_8ae739bbdb2e29ae1b84ed5f7a6306e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.499579668045044]]], [[[1.462172031402588]]], [[[1.4842655658721924]]], [[[1.1194058656692505]]], [[[0.9996048808097839]]], [[[1.6772676706314087]]], [[[1.2080305814743042]]], [[[1.1058160066604614]]], [[[1.2655586004257202]]], [[[1.5411800146102905]]], [[[1.372831106185913]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_651a19819dedc4b7d3473fa9edacca00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4c6300180eb71b9894ee6150b7682ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d42eee4fcc67368ca8930baa65141d58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([5556, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_651a19819dedc4b7d3473fa9edacca00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95b1ca913207ec78681cc8bca7c5def7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([1744, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_316e84b1b2f50c828701b159b860efd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([1547, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d03fba44a0cfc7f497539516ee04c33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.764944314956665]]], [[[1.1515692472457886]]], [[[1.0472049713134766]]], [[[1.9638758897781372]]], [[[1.9510688781738281]]], [[[1.4468040466308594]]], [[[1.6598985195159912]]], [[[1.0138143301010132]]], [[[1.0680865049362183]]], [[[1.635872483253479]]], [[[1.2494542598724365]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_1106d74124618a347239729e4487f08a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([2056, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d075097b0953ca67a604f32ab994b49d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57cbd70a951d7b2ed3bf8eb46a8eba03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([4650, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8978058a96f9fe358665e91453d65d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([1059, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b09a3f5ffdbeef477cc04f0f16a8b6b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af462d40151a42f97e6f9748d978600b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([2347, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7a6de004225198b2f0c15e19863a65f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([3109, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31d97c96a53121acdb15f687dbc6c6e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([3813, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_400d5410518832a83bc355c6d829d4fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c13d84838b3dd5e55457c963cc0bbda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.2873609066009521]]], [[[1.6012513637542725]]], [[[1.6298158168792725]]], [[[1.5765769481658936]]], [[[1.364870309829712]]], [[[1.5271124839782715]]], [[[1.7190725803375244]]], [[[1.0701406002044678]]], [[[1.035871982574463]]], [[[1.7909011840820312]]], [[[1.0524457693099976]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_651a19819dedc4b7d3473fa9edacca00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_68011cd8372efb68dc7c0b02ae69e72b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([2100, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76be76b17c897da0fdfdc0586ec39e00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d9bd6a3d984d093d3f9ce2eb286aef1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([4231, 4], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()