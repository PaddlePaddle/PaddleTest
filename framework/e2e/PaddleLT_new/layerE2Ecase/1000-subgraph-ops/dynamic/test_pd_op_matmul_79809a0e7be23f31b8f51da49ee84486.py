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



class PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc44b8822fc80d2b6476f9ef4656e4ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_18974c795569df05df5c9d63598e9bad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ec382d49d74aa6d89dff010e539947cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1cce7d797df58487b0060bb833d1044a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 32, 100], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bdb76f386eb4f20b7dbdeb850fb04b12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5cf4a1128358f589ac6ccb8c4ac81c59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f470a46364c9842a9ff0637565751431(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de52dfb585380e2d279ec5db4b96cc89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 4, 7, 7, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c7376be36727addbc3b9af8d16852d6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be0cd65e25e9e490f24e721f78dd6f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 64, 1025], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f42a7131f4906bdb1cea27783b080ddd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 1025, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_92682107d79f6b9f43e23273a1debfe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_de52dfb585380e2d279ec5db4b96cc89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 4, 7, 7, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2da82bb5b434b66cd254ddb5288773da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0a5ecf8762bf63d510ddf7cdee5162f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_135fa146fa70acb4e2b049ad9c3d8c0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 168], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1e6c9e0c72fb62c93df506fec0da9e80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([168, 672], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e69fbeb40bd803e63b96a9e543f04ec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 4, 7, 7, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_016830cc7c44256fd7bb2462e5b2590b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d54a94b4c784c6d16df1dd2849f460d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d2fec1d5f5a4d9b642e3cf89ec432d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 32, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3ede8368129c06cdd8e003be7c5d14a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 1024, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_016830cc7c44256fd7bb2462e5b2590b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8c9cb08b5b4274244c7d79ac19cb467f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_838d2c88eeb24044e876491970c74603(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6399b6c9c56a24b7c234bac9300736ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9d58f1a7fbdc1cf50e418880bbae2f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([11, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([320, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ae10ad29aacf11d4942fb37c44d9c8fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_12963c54b8af53d01339f0f26cf88b73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ba1620cb692388dfb88a39e626aa9019(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c6b53632468226a1460cc37aafc35256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 3, 64, 197], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c3713c7ea9915f5ec60a6c7bda3deabe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 197], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_afbe1d675a7b09b76a1b7a7e116d9b48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_64fe4c989ef1b2b81154b5005dc9c6d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 2, 7, 7, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee759f1154d48543e10805355326c2a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 2, 7, 7, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0b4e2862999eeb7d71b1a2331752d201(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_606eff933fd6ac764ab018e08194468f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_793bae24e24eb5209ca1970251f0fe3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d0cedaf8999644eb3b30af6a79613feb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 2, 32, 640], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e132416820e3eacaecda2197f0b9c32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 640], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1d9359a677102d19bc362a8adf33118f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0b4e2862999eeb7d71b1a2331752d201(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_606eff933fd6ac764ab018e08194468f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6fe567beb44d4808b4cb5544cdb58445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b2243d8179d9345a86020c9a3f9958a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_25306291e7d545ba51a506a654742246(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 8, 7, 7, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5028e8356dc2fc5cb71cca7a28af5441(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2111307a22072a59aad15c5ac7a3b576(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9d1988b67c1f57b8563621903dabb570(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0930c836ac267896c138132a18cabeee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1988b67c1f57b8563621903dabb570
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 16384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 16384, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_01c8d9813819f25d431573d274c9be38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_44f9eee65793203d41f4810d4279fa39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 2, 32, 200], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e369eacd3c749bd7d3f25bd98d260f09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f17474b1d3684b517b66dc1994b3fac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eec2551baeaff79067a3f006a53a1bff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d04bf4a7ca68945dacacd1d1b0a4b0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e6665a2e68937e5022ca84fac94c109a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_054fda6cb12c76dde388abdce4281099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 1536], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1536, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be344fac259fbb5fafb95e58e36afacb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5489b02ddfbe863f78bebb55f17abc4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c03827e3472d19ef4f405c4b3ac843fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c085763b4f2bb2b132af07dbcb6f30a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f29ea875e2c28819a2a1c0cbfdf69fc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c7a1b759928816e1dac70687beaf43ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9961228686b4451770f256696bd2e1f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1d7a899062e2499b73c4a17f770e46cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eb9d88d4613013c03e56e31f3ad3fa1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9750a08c3a6fa12cc08a5a8c0d430821(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1988b67c1f57b8563621903dabb570
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4096], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4096, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cab0f83cd66e03d4fd52585ec234eff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1988b67c1f57b8563621903dabb570
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1eb46ce7d355b44d73d7a56f7b242155(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_26b9aa37c242bfa20aa40ba81890f5f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1eb46ce7d355b44d73d7a56f7b242155
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 1, 91], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([91], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b819d5413ade6c6ef0fd7fcd7d23860f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([4312, 16, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 48], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d31bab3046b0374c39625d56f35da99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([4312, 16, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9c87e89d1be2e86af61e4c8b18adae84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 1, 7, 7, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_600ce1c9d9e6fc82f87de63a22918685(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 40, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 6625], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_18aa6a0a09708f74a110ee53541a3467(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7620ab0653a1baf06860a5deeeaa443a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 8, 32, 160], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_29280c7a62dff54866966898f6b96f08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7cb970269313eb0fe131611f2f36f273(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b5e767a53a4859fe37b9ccd53388b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 960], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([960, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fe3283c04acf628bdf09e405b8a6d41e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 960], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5309c95341dd9da774691d25f93e670e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1024, 2048], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4bb2ae438b499f95f3e1811e3911de4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2048, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0bc929032b2c1f2ee058bda2371ad4e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 624], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([624, 156], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bf24c6788b0a1dce52faea58783373fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([156, 624], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e69fbeb40bd803e63b96a9e543f04ec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 4, 7, 7, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b90e523f026e03f1d4aa7dea4db8ef1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f4dbef54087c6567e52d346f5be22064(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3290affa202588bd4b03907ea40cb068(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e90dd707f7bbbac5cef2c0a1d0f90c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fcec9d9b6d6b166dc0714beab4ded196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ded61549b6dfbd52511e092992ae206(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 32, 320], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5ceeacc69d6be4253143d9076e0eca1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f260f156e017fb6e065e409ea9967c23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2890e8912570622d5b1ca01b4489707e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_152420043be3e4cb63fbd17a930d471c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5220883f2f95d54bebb47da6669d2614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9e580b3c412386ec09fa6cc9d003acd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3290affa202588bd4b03907ea40cb068(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e90dd707f7bbbac5cef2c0a1d0f90c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10734c0b094e09c8e8a68ac7aa626201(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([390, 3136], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3136, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b7426bef8031fc0cb3d281edc1e9fa23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1024, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4a14b23e2efdc2311796b9791fadcbd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2bf7e820b03f8e524ed16fcc1e93dbd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ff8c66c03fe9fb229c76f1f2095a70c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 32, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0b6a01dcd783cea716d09820bbbe99d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 512, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4a14b23e2efdc2311796b9791fadcbd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e34e6f25698260fbb62749eed8bb0b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2e5c226effdb4aac1e4f72a0dab633d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 72], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([72, 18], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b7c8ca8db6a047ded3d16007f354a7e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3876880705356598, 0.8070262670516968, 0.0, 0.0, 0.0, 0.9655231833457947, 0.0, 0.0, 0.02768191695213318, 0.0, 0.5215060114860535, 0.0, 0.5745652318000793, 1.316960096359253, 0.9228256344795227, 0.0, 0.3434208035469055, 0.6145989894866943]], dtype='float32').reshape([1, 18]),
            paddle.uniform([18, 72], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e2b5405896cba4980abe60fd84abce55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([11, 704], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([704, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4fe14453434b3b18b75ab0f1ab3cd0b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab83bd9e5c6b8e2679cc4ce333837ac6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 3, 64, 198], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_37b06a24bddf16fb56b9505afb8764ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 198], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0d2596007a72ea89070c611747d14ff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_784f1e339ddd277a9c2a029b9cbc001b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f47c3204af062793760f5f3b6c79ea8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_32c9212148894626be8c1602a8875a99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([11, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_adcb00e165d80435aea8f6fb3e6d3a4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b3fe429cf9bf0f5fef5234bf022dd31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_306fa518b052d036906bc46580425684(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 64, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ff633b349f9af58e4ae539a34b002f00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 1024, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_adcb00e165d80435aea8f6fb3e6d3a4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e065c62c4798c1a7113cc2d7b4bb0874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8439f90cac4604ecabd97cfe77a2a9fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eec2551baeaff79067a3f006a53a1bff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d04bf4a7ca68945dacacd1d1b0a4b0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cc0fcf2109ec4e714728bb567e1762a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1988b67c1f57b8563621903dabb570
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 32768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 32768, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0433df30520f317edd200dfb8d064fe9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9699efa9de513ef730cfe0b629d736da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab746372301ba2eec31f2ec11962fcc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c94eb23f6bf88cf7c0e9b77d975cc9e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 8, 7, 7, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a5192a400929800e0b6f5fbf33b8b387(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4d78f8cad2f7ede4a783c270334d160(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ae2b5f5b41178745856447ffb89bbb4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_31d3ac93886f83e745eee9d048cc0340(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([22, 197, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7fc083e962b681230dd5ce6e0370489d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([22, 197, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_da1e31bb1d0f423dbcc9fd7db6ceea10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6c49bf5e270257ad6f7e41ef3f66550d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 64, 1174], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_85a992bf002aba78535fb01683d18f45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 1174, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9d9e68303d2f650346e044543481fc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_634fa7edad1af6aac52956b1fd0774e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([43, 704], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([704, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a8f75d3b7985c87cb59d501d8d3f2ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1eda2d880b0d72fc3432839f63059938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_02a43375031e4229e8fa6139ad9bf8fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([43, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([320, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a01d34837021ef27a23d3fc06649c03a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bff49c09efb1d209181847fafd65db5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 3, 64, 198], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4fc46487177c3e220674be16889b672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 198], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_44b5c0081a16b3e473a39a6bd3f995a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ba4877acc856269274752a953ba3df9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f3842e228bd2218f85c0102d8aad44e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2e9e9712cab09f746d3a31545468a5db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b7efbfa7e08635906bf5ed55fa58287d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1988b67c1f57b8563621903dabb570
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 64, 8192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7fd220baeaebd60bd5bca1a18731f7ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1988b67c1f57b8563621903dabb570
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8192, 8192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6d892db3a6833ebc2cd32492dd7a2f6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 1536], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1536, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dc44b8822fc80d2b6476f9ef4656e4ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_18974c795569df05df5c9d63598e9bad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1d7a899062e2499b73c4a17f770e46cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eb9d88d4613013c03e56e31f3ad3fa1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54c5fe76e613aac62daba38a21693ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e1d113a068045e7a02153554b2206d00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 8, 32, 50], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cc9e73ee407b54dad0c49fc3f69f4d7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 50], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1cd74943cdb13358287c2a6c1f7252aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a141c8ff1ca3c489a9db22918b9e1da3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([43, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e46c44c6e1064534bec86b9d9c9cd52e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([512, 12544], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([12544, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b41327f5cf1015bd97f4feb0b00522cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1024, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_376cad60b6dc789c59900914212762d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 7, 7, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8c511fbfa8a9f9282c02891ae7425cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9ef2fcc6d21ae13dd53f72a840af53c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_92093643eaa890dabfcbbe036adca6bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6, 64, 1025], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b7e501afffc91a16e750026203b977a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6, 1025, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ee205ee5937f81628750a7cf79783b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9c87e89d1be2e86af61e4c8b18adae84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 1, 7, 7, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee759f1154d48543e10805355326c2a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 2, 7, 7, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9699efa9de513ef730cfe0b629d736da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab746372301ba2eec31f2ec11962fcc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1aba11eb7080911f70c50aebd92a6963(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_436f35f9488fbc8bb1111ff9b404080e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_62d1e9639a1b6a268d6317870809d595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 3, 64, 197], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2dcf4f1a171c5f2f9df7a4cb4b8816db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 197], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0f70ea740ce7248b6321d3982d336a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1aba11eb7080911f70c50aebd92a6963(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_64819b994d8fc8a755c6d6f25d1ee794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1988b67c1f57b8563621903dabb570
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 64, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d5db795bbad41a5eb0894416ec7ce3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1988b67c1f57b8563621903dabb570
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4096], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 4096, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_64fe4c989ef1b2b81154b5005dc9c6d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 2, 7, 7, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_135fa146fa70acb4e2b049ad9c3d8c0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([672, 168], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1e6c9e0c72fb62c93df506fec0da9e80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([168, 672], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9c0da5b084012c0de48e86a61e9094c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([43, 1280], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1280, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_782f6d384ff84659415767db53656658(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 197, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fd171aaa20b4a9f842210deda54fd9c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 197, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_32b60454b62fececa3565cdc5f0e896f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7ec7d1f8210889ed416a53c2fc363673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 64, 577], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c37bfeed16e407f6fc3a21eabfda5758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 12, 577, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e65729111ae64e5ada80db1f1726fde8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_25306291e7d545ba51a506a654742246(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 8, 7, 7, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0570a1e3b7eaddb9f76d50afaa5f58da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([156, 39], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_73999dd53b9dbe3781e0b8ba68fe38b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 39], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([39, 156], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_66e6f6d886437cdae320dcfa894f0509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1988b67c1f57b8563621903dabb570
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8192, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_968b8b6b0cffc82fa03f627ae83d24c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d1988b67c1f57b8563621903dabb570
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 8192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6fe567beb44d4808b4cb5544cdb58445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b2243d8179d9345a86020c9a3f9958a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1aba11eb7080911f70c50aebd92a6963(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c94eb23f6bf88cf7c0e9b77d975cc9e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 8, 7, 7, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_470941b5181f2e7cdd78149ff9a98abe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2862354d9fd8c74eec163dd9ac55bcc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_98eb9c6eba3c3bb1a2baf96dadd79888(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e06e39613d40b4e4ebdae25b1f9a958c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([872, 218], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_30cd1da48f6ed8b67401b87020be835c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([218, 872], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5489b02ddfbe863f78bebb55f17abc4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c03827e3472d19ef4f405c4b3ac843fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e06e39613d40b4e4ebdae25b1f9a958c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([872, 218], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_30cd1da48f6ed8b67401b87020be835c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([218, 872], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_097064e9100185da1e687fbac3b63cd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([86, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_097064e9100185da1e687fbac3b63cd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([86, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5664a3193b63908791a4b967ce209119(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([92, 23], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c47c2cda68b7f96e777e54352d3f3b49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.6389513611793518, 0.4143849015235901, 1.1890809535980225, 0.0, 0.0, 1.0791393518447876, 0.24445515871047974, 0.0, 0.0, 0.0, 0.8210386037826538, 0.4099501371383667, 0.0, 0.0, 0.05812513828277588, 0.30904722213745117, 0.0, 0.0, 1.5841295719146729, 0.0, 0.0, 0.0, 0.18904435634613037]], dtype='float32').reshape([1, 23]),
            paddle.uniform([23, 92], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_794d6b641070cc68309ec9cb05b49433(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc85d4f72f185f0a4586a9bed05afcef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_96ce0f80a0ff4db13b46f78e88c0e917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bf7dc8466d40ed0d0d737ddedaca8536(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fcec9d9b6d6b166dc0714beab4ded196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5ded61549b6dfbd52511e092992ae206(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 32, 320], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5ceeacc69d6be4253143d9076e0eca1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f260f156e017fb6e065e409ea9967c23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f46544d097536d29d010a8141f7a7f65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_242ffb89836a7bd896ad3505f0598a5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_57ed80701803de3aee43b4a41b771e6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8b796614f0cf382df9925abc249d8e49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 64, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1c9189cf403a3f5a26fb96351c6b5d08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 8, 512, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_242ffb89836a7bd896ad3505f0598a5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2798ad02f9b5c5079eb839ee9030b551(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([54, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2798ad02f9b5c5079eb839ee9030b551(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([54, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_794d6b641070cc68309ec9cb05b49433(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc85d4f72f185f0a4586a9bed05afcef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1e3526733b6d8a46cdb1a870206eb10f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_852074401fd4288e2ee3a83ce8d0303e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ec382d49d74aa6d89dff010e539947cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1cce7d797df58487b0060bb833d1044a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 32, 100], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bdb76f386eb4f20b7dbdeb850fb04b12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5cf4a1128358f589ac6ccb8c4ac81c59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_034420f9de4cfd6350ea30377996323c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1248, 312], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_20f4b6b6010ade5151ad5a497b1f73f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 312], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([312, 1248], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d4bc52ff00829e0df20310203344606e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([11, 1280], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1280, 1000], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4bc0f50a3594d6773888a18866a258f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 30], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1c43e6abe19a0dec904cf45d08e3c050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.058657944202423096, 0.44569411873817444, 1.4558477401733398, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.7536977529525757, 0.44441232085227966, 0.10664764046669006, 0.0, 1.1139302253723145, 0.7939410209655762, 0.6237434148788452, 0.4778575003147125, 0.2416335940361023, 0.22969874739646912, 1.1675066947937012, 0.768220067024231, 0.7505300045013428, 0.3666670620441437, 0.0, 0.25254523754119873]], dtype='float32').reshape([1, 30]),
            paddle.uniform([30, 120], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_de9b2753729e9dbafbcc68a13bee6b11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bafde56376eb6dbe4b4493879f3cdf60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6, 64, 1174], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fb13d3c67091b76f935713ee1ca59462(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 6, 1174, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_196c4a6a0480c2b97c97fe16f1ba4525(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c085763b4f2bb2b132af07dbcb6f30a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f29ea875e2c28819a2a1c0cbfdf69fc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a22a69c6159f49ca21b97116e3b67ffc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0bc0101b8c26a4e4bf2ebb0d5ba0a4e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b291d5a6a9d388ed91e27baf6b68abe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_37a703401ebfd81e4e8412fced7ad99b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4f6b22f0a138502f7a44edbf0bacf5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b291d5a6a9d388ed91e27baf6b68abe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_37a703401ebfd81e4e8412fced7ad99b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a5192a400929800e0b6f5fbf33b8b387(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4d78f8cad2f7ede4a783c270334d160(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f7535dd5989757347a558503915a9587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1960, 16, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 48], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bfb64ad3c4ca1b5f7f24b4e8622c3dec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1960, 16, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([24, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f77e836cad43f819d1375ef06df40f77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 480], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d0f3b3f92a65ec27ad0bcd63faf6d92f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_db9502688260ab70cc2c40b3228e2c5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 25, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192, 37], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_376cad60b6dc789c59900914212762d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 7, 7, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()