import os
os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
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


class TestPrimitiveOp_d36dc4b118a182055568ee1fde7cbc1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([1524, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9a98ca02c408970d8e404d04c1aaaabf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([2340, 4], dtype='float32', min=-0.5, max=0.5),
        ]


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


class TestPrimitiveOp_ee1975b6492c643277609cec7be5dfc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee1975b6492c643277609cec7be5dfc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee1975b6492c643277609cec7be5dfc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9fe2323eea1e183f5cbd7dd776f34137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.6020722389221191]]], [[[1.485905408859253]]], [[[1.2452826499938965]]], [[[1.7729523181915283]]], [[[1.7634717226028442]]], [[[1.197999119758606]]], [[[0.9173964262008667]]], [[[1.8267507553100586]]], [[[1.414074420928955]]], [[[1.196136713027954]]], [[[1.648266315460205]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_2bf0691a3c0e3a92fb7a3b70c5f27a6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee1975b6492c643277609cec7be5dfc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee1975b6492c643277609cec7be5dfc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4495ab5f4be728f0792308fff458af3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([2047, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5b9ab49e2324dada2810ac480881e090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.1757361888885498]]], [[[0.9348726272583008]]], [[[1.2149393558502197]]], [[[1.3885247707366943]]], [[[1.4686007499694824]]], [[[1.33112370967865]]], [[[1.8724888563156128]]], [[[1.557882308959961]]], [[[1.0384336709976196]]], [[[1.651749610900879]]], [[[1.494024634361267]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_49ac22e3891f0b67a0eea0a2e7b71811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([1813, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_84123b34f47af12016439ef6e550d401(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([3061, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_677becf330ce7f34c3a02f5e587f2c46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d16900c8078c4982ff57b87df281bd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8f711c123bbd8b919a99cb08dd060f66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4964e6692c8c9792eeca25935b2cf1fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3516240119934082]]], [[[1.845107078552246]]], [[[1.7896212339401245]]], [[[1.3700904846191406]]], [[[1.1715631484985352]]], [[[1.0827648639678955]]], [[[1.588724136352539]]], [[[1.8259875774383545]]], [[[1.4885084629058838]]], [[[1.5828368663787842]]], [[[1.9075171947479248]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_7d43760643aa635d1a8d049234309aa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([2062, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_13f61c0aabecdb3c20f8d76c30b81638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([5526, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b704e667d9730eb3c4022794e8ae71f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.7837913036346436]]], [[[1.3828532695770264]]], [[[1.5317630767822266]]], [[[1.665555477142334]]], [[[1.6844282150268555]]], [[[1.2563235759735107]]], [[[1.1857061386108398]]], [[[1.3448436260223389]]], [[[1.9404804706573486]]], [[[1.2835737466812134]]], [[[1.0105714797973633]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_9882b89b41f5270c025c7377cf0983c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.8084135055541992]]], [[[1.0968878269195557]]], [[[1.7551380395889282]]], [[[1.3578224182128906]]], [[[1.0439553260803223]]], [[[1.429886817932129]]], [[[1.5221811532974243]]], [[[1.0403099060058594]]], [[[1.2404347658157349]]], [[[1.097940444946289]]], [[[1.8083899021148682]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_924c396f8dab8c85c1b974199dcea4c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([1071, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_92469a9f15f86f9e590cc0987ec9835e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([1760, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fda5e1ed2e1c7ae9fb8b0b927fa58aa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_463abca34f485cc923eabecd39128569
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c5dfd405bbffa9afad7a0127ac1bb3c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([4204, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_68a520c38babd3536d694c0a1ee21c64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([4680, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00b30e73629dd22da5ad71c6304471df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4d3cbab577d074dccd5e86586288195
    def get_inputs(self):
        return [
            paddle.uniform([3778, 4], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()