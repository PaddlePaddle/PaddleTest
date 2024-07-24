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


class TestPrimitiveOp_e54e094c1d02de3474f11580c10dc3c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3794597387313843]]], [[[1.6143207550048828]]], [[[1.6783286333084106]]], [[[1.7263753414154053]]], [[[1.747666358947754]]], [[[1.0856425762176514]]], [[[1.5064505338668823]]], [[[0.8724386692047119]]], [[[1.3799586296081543]]], [[[0.9480482339859009]]], [[[1.6101467609405518]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


class TestPrimitiveOp_4b66e55a467fead77abbaf2d1b3c36dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([1762, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d0b1ef5bb13abdfbc246311fe443df8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.8708384037017822]]], [[[1.6472737789154053]]], [[[0.9556787610054016]]], [[[1.4333670139312744]]], [[[1.816613793373108]]], [[[1.2200453281402588]]], [[[1.8740272521972656]]], [[[1.5666255950927734]]], [[[0.996346116065979]]], [[[1.0640770196914673]]], [[[1.3106660842895508]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_ee72b84d5267abcb7ac848be8ee4a5a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.4013986587524414]]], [[[1.4842860698699951]]], [[[1.4070405960083008]]], [[[1.9098567962646484]]], [[[1.7554945945739746]]], [[[1.9352868795394897]]], [[[1.199039340019226]]], [[[1.6022703647613525]]], [[[1.423989176750183]]], [[[1.6983251571655273]]], [[[1.3227473497390747]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


class TestPrimitiveOp_1d51813679114ed0d4c2856129e075b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([5522, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_651a19819dedc4b7d3473fa9edacca00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_309898a0ef2bff0264e9a133f3df564d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([1760, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6e9b87f0d5dd9c6a3dd32f72e17e10b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([1522, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d8108bf88eb76bfaedb379b82ab8249(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.8364286422729492]]], [[[1.2665343284606934]]], [[[1.3048250675201416]]], [[[1.7090530395507812]]], [[[1.3789656162261963]]], [[[1.7982077598571777]]], [[[1.0092480182647705]]], [[[1.8642301559448242]]], [[[1.3640177249908447]]], [[[1.3980094194412231]]], [[[1.9695626497268677]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_f5aaec55a6bb34084384542ab0fc9469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([2074, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d075097b0953ca67a604f32ab994b49d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_059ba5281c3cb4ed4073c06859bbf9c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([4734, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a76c0f80467632b9fc7414fab57acc04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([1074, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b09a3f5ffdbeef477cc04f0f16a8b6b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_956d4ab3888506c52e454923b24e5e7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([2332, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b9e3bba9ad25b065599e5ed8a6721ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([3051, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a2a2184d44df1a76cc716d7a365ec1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([3870, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_400d5410518832a83bc355c6d829d4fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39914d77b3b944251ea17d37c9fd880e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0129414796829224]]], [[[0.9902558326721191]]], [[[0.996795117855072]]], [[[1.5246710777282715]]], [[[1.057382583618164]]], [[[1.4440983533859253]]], [[[1.6624338626861572]]], [[[1.5157499313354492]]], [[[1.5402613878250122]]], [[[1.8550347089767456]]], [[[1.0058176517486572]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_651a19819dedc4b7d3473fa9edacca00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_212cd086ee4719ab98090deea437eb4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([2111, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76be76b17c897da0fdfdc0586ec39e00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f612e4d5de19d4caf08cac387b39bd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([4141, 4], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()