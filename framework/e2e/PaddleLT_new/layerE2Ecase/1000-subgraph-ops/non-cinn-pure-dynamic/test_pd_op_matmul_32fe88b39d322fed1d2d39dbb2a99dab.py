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


class TestPrimitiveOp_ea7fbb677ab351c5f2ee627182f1dce8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1eb46ce7d355b44d73d7a56f7b242155
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 1, 91], dtype='float32', min=0, max=0.5),
            paddle.uniform([91], dtype='float32', min=0, max=0.5),
        ]


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


class TestPrimitiveOp_5237b85f8ffaeba738f550e7efa70b60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44883c2f351da7436970e1fe2a58bdf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_2c7e2d4a4abaf3c9c1262c7cadf5c014(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_da808c997b1606f9abbf5d3f00a60fb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([72, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_12c2b70e283ec3b1d1d18928a69769d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.to_tensor([[5.034132957458496, 5.051319122314453, 5.2611212730407715, 4.580645561218262, 4.762973308563232, 5.002790451049805, 5.6654815673828125, 4.349248886108398, 5.136642932891846, 5.052401065826416, 4.996928691864014, 4.306968688964844, 4.845163345336914, 4.200080394744873, 5.610182285308838, 4.703643798828125, 4.884689807891846, 4.9336838722229]], dtype='float32').reshape([1, 18]),
            paddle.uniform([18, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8dc38aa7b61c20e0187138d51f9c84b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_bc8b2f4a3758c180ef4eb2c3469c65bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 32, 100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d8e45836b9ba07d82da6981c36106ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_220f62c2be5938415330f0f8b41c04ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e01627652252892028bede2a144b1d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([92, 23], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe8cfeab7e541b942d094a62740bfd3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.to_tensor([[7.229928970336914, 6.533698081970215, 6.732686519622803, 6.243114948272705, 7.0304975509643555, 6.9293904304504395, 6.767514228820801, 6.359036445617676, 5.74185037612915, 6.8545637130737305, 6.183698654174805, 6.503045558929443, 5.956207752227783, 6.541355133056641, 5.515870571136475, 5.586697101593018, 6.758831977844238, 7.21973991394043, 6.840029716491699, 6.680665016174316, 6.361820697784424, 6.038775444030762, 6.562516689300537]], dtype='float32').reshape([1, 23]),
            paddle.uniform([23, 92], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_789634c9d9743b1a4f84fd6b33d2ec64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c407f750e7a2af845ca60d975d32fa86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([54, 3, 64, 198], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d7bec56f61c96934bcbb9399eb09e47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 198], dtype='float32', min=0, max=0.5),
            paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2c2b3ace8ab360e67ae26bd3a97da7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_542605ed7f645e434e98c19ebda0d6cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1960, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4177226450b23e485197a84cd0cc434c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1960, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95c6505fa2d217279961f8011cd7f982(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c487478543298b3263113c654d090db3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f473e93aad6498276dbc9542fdfe8ccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d9ecd078db37185f1bc9df7d1fb9d0a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
            paddle.uniform([960, 240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_50993ed45b83401973a2b88e5fc6791d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 960], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dccba59f471038d9a136fb746d1b63d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2fd2b381bb65ee3e23c5399b46959821(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d500c06b9f19724aa9e1ea965f28d117(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([512, 12544], dtype='float32', min=0, max=0.5),
            paddle.uniform([12544, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fbde91179b63da6a327b4b21019a6b4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1428773ac25be2fe0044c9b7a8cc630(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5f725716ae15561f333a7dffa3e3cfc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d423c158814448b910f0de789cf4a66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_964f834ec36dc986894fa6068b5596d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffc7be1808b06b37bdd8f818290be0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_149cc8cb6bb06fb12ba654d760dc6616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1cb5bf10398409be27f58713ecb20fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_33a0eac6f0a241a8675e28fadee53468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_00ea59c576e6a39e5171bd710e0bca67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06511155eaa36d78dd56ebec5f55b943(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9456bea98049755ef90294ca269c559b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([11, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_00ea59c576e6a39e5171bd710e0bca67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06511155eaa36d78dd56ebec5f55b943(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_257552732e1918673036be3a10508a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd6f2397cf16046541fbb39481a5d60d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9208fa49a1d41fc762d01a9ba8332d69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0c0e93b406184e92008464d68c122ae7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 32, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3cd5e12a86f28291d1a19237b5bf051c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d30b37063d3c5fae11392899cfe2a037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4906f4e5fc94b06171b9c001194a4178(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6001f33a896d2f0076c760bf044064a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5cbec6528720d10f12b8fe64835482a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8013e70993ff957377712b66d2164dd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4fed18178d25b49f0f01d3c91487204(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12, 64, 577], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc2b822a939fc1e53e18648111787252(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd626bfae217279d927bcf195a07e5e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fbc088399a7a7ca100e3275a893f925e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1f6e30c2ccf97946c973f74239cdae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b2739c8f327f344e4a5884e8d890d9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 197, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dbc8e7b5603a24febe5c9e47674002a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 197, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf85beab0df7ca8be73cd0e87ebe9317(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
            paddle.uniform([872, 218], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfd6c227bc326c9cfe66082617440bb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
            paddle.uniform([218, 872], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3028ac2e4407fff6ca04dc4a15fc2715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_50c69b800e36b63bbe4421c3d2647897(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d633a87913342ef0bd39325b9be3d10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fbce92414dc9c913b4212c9249001850(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 64, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3fa22e32aff8a8d82ce96279a048b73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_50c69b800e36b63bbe4421c3d2647897(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_257552732e1918673036be3a10508a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd6f2397cf16046541fbb39481a5d60d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7d289397ba34502fe26253d1b8270fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fa01a146bfc60ee9bdd207ca3b78fcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b73310187cccdf75f7a0fa02cfb63636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([390, 3136], dtype='float32', min=0, max=0.5),
            paddle.uniform([3136, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af9483a3a26eaae9a8e45bfefa865742(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8e01dd045717a69396c4bd50614792(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b38d85daa01576623d01b82309e6b9ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0307c68c9c050499ce67ac0c4d1dfc12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2561418c04154b01169d09c3ab1c12df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 2, 32, 640], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecfa58423d887dcc805af50458a5dd41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 640], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a1981cca16b06c0f82795944a4b2381(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7daf027e57d0062166bc9faea9c57498(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_243266818017cd19102f5d4969e5c5b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11b70e6c7b69422fe272d962da010083(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ca5da050e1ac1dc8bf7e9dd454c8d74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([86, 3, 64, 198], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a6176973253f05aaa508e79d826ccf3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 198], dtype='float32', min=0, max=0.5),
            paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7668d41e50b5a9dd4caaea9dce04ad4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d952bf8903f37f860866f36237add0a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8db2d3a5c75c3bafd68869c08b25a79d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06d4a12dbeffc3050fe8fbdf68eb3734(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([86, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06d4a12dbeffc3050fe8fbdf68eb3734(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([86, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_059444db915cbe26a2cb38c770848fff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([11, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5f01ab1ac9d15505bd5831bc91746eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([54, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5f01ab1ac9d15505bd5831bc91746eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([54, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_571bfd749d37d359a1d4876bf4303ac0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 2048], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9e0bdace653f3a2a6763eabbb781090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ed2081bc06f9459ffc725c6e41c27a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9bfa30a0377aa8c6ade8d0a88e6dbb90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([4312, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dff539d980adfedda420296e889cbd2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([4312, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1428773ac25be2fe0044c9b7a8cc630(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5f725716ae15561f333a7dffa3e3cfc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d423c158814448b910f0de789cf4a66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd097b929ec25d7c8111082f0bf552ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([43, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7c7e7343a5c3335f1274c76958d17277(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed2a2fc89934383d72f90f0038b2d4c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5289b4ce787f22e8465183373be5f1ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa6f5d2a621a8af938ca5b74da0e5d3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_018d842aa3cf2cda4e55d0f13a2f5d27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 32, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6aa556e5567dfdd98d2c2f1b8b547c6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5289b4ce787f22e8465183373be5f1ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e012c17b4860daf2300bb6446d7a9d44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([43, 1280], dtype='float32', min=0, max=0.5),
            paddle.uniform([1280, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c51653b028c96d893e1bf0e7c1daf5a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4e9565e6aa84dece5d43f5554e44dc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8dc38aa7b61c20e0187138d51f9c84b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc8b2f4a3758c180ef4eb2c3469c65bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 32, 100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d8e45836b9ba07d82da6981c36106ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_220f62c2be5938415330f0f8b41c04ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e961338b6e7309928e19330a32336c0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7a6d5e7af796605175acafc004bcc9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4785dab63516714eaaa93b886561061c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e25ca21eb62e5b61d5dd271b98b310ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_baa3e10925de79c7e3e5a908ca7bac26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce292ed5e27d2b77fee8c5f4d25f1d4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2b7c8b86e62d094eeb492762cadf27fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc8e01dd045717a69396c4bd50614792(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b38d85daa01576623d01b82309e6b9ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8a50417092459bb11344395a75ae0a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 1536], dtype='float32', min=0, max=0.5),
            paddle.uniform([1536, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a5140581d6cdd5a0b2a4fa45da658117(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f1e05a4651db4466e33b12b2949d7ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_437d429725f9aefc7c58bb3f7c851fc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_397302bc1bc9170118d025716103a7d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27d2bdaf3dda97103397d3ce61dfa5c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 1536], dtype='float32', min=0, max=0.5),
            paddle.uniform([1536, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3028ac2e4407fff6ca04dc4a15fc2715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95c6505fa2d217279961f8011cd7f982(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20becf49007809ca8299afabf418bbcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c86eb0112806408f386b4b1aa220ee6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6, 64, 1025], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_26552298659c31a383f776719e5ce6cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92e424021650cc5ac65a60f07d420bb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9be8056f92ebb85842f65d1e7d308e55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbaca0b7f272645517d3bacbb47a83f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18f07e2dc15a11bc6eac5f7602806b97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6bdbf7dec4b389b5280123936b0357d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ac3d21ec31c1a8eb97f875b4f34e5bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 150], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db8b145211f0daed048d90b0a4e447c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e5b6839da18a5c4e2e242a18772aa62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5407da3b1b60111062b71f8efe08b4d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5684a5518f007c04bb8966cd363c8932(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83c9577951e418c1f18e3e1782797dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2e4d6364ce20d823401e3f5053a67e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec2a7526b95a68948be5b3b5e50bce76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9296691d35216b0061c1a4b0fe47c7b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 32, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eb557ff11cabfb4fd9c4b2e10fd20339(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2e4d6364ce20d823401e3f5053a67e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c85bab461a8c29da0e66e1e87433bfda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb0e6d44df3a1f18e1dc1a392792646e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee07e68c27c2a86fd55de9f9694b3041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 32, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1158544d23ce09beff5d1a70ccb9e151(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c85bab461a8c29da0e66e1e87433bfda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e25ca21eb62e5b61d5dd271b98b310ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_baa3e10925de79c7e3e5a908ca7bac26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5237b85f8ffaeba738f550e7efa70b60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44883c2f351da7436970e1fe2a58bdf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67b3df1ad0bcf36f3333d86b28f49cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([10, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95c6505fa2d217279961f8011cd7f982(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfccf58ddb43b2f8456bff22959666b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([43, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ed2081bc06f9459ffc725c6e41c27a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d473dad628c5aec767438580a2593078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_be40e062abb5f14d487c16c7f2cc0b27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([54, 3, 64, 197], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e29264f028df1264cd7fa1f5e2d6f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 197], dtype='float32', min=0, max=0.5),
            paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2221eee62ae3b8180fe4560fe85e5658(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e72435a1be8a999d48f7ccc6ab0a7ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60e003050d91d00b6e0b75e86b79f9a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ab00e8cd77cd283f9707e1b2abd6299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75c741fe4c9e50e13c98baa372780c10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e72435a1be8a999d48f7ccc6ab0a7ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44fc3dd1a9d511e545fc54c0d12ea679(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5cbec6528720d10f12b8fe64835482a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d952bf8903f37f860866f36237add0a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8db2d3a5c75c3bafd68869c08b25a79d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_985af7d27894b254f46c40a638473d60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 40, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 6625], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9208fa49a1d41fc762d01a9ba8332d69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0c0e93b406184e92008464d68c122ae7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 32, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3cd5e12a86f28291d1a19237b5bf051c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d30b37063d3c5fae11392899cfe2a037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_209c64c51b52b0ce5e604be991255658(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6fe48ec2f4ca2f9c40d6c4db0afe6e13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd94d6f65097c0cbd37997555ac7273a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 64, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e85f6b4a5aed395bc96fd242b103af5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_209c64c51b52b0ce5e604be991255658(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18f07e2dc15a11bc6eac5f7602806b97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6bdbf7dec4b389b5280123936b0357d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd1e3e498031f4c8f86fc5a5620f0814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 150], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_040ef44e6ad441ae8f4daffa403643d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7fbc36b61f9295af22c389535783014b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 2, 32, 200], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a6dcd06080eab46ae850011ba431f6c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2fa0deaef8a927ce24e0c0e0a0885bcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_170d3b1a85adf5197056248a5e7a215d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4039d448a2f90a485980a53a9bef7fc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([22, 197, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_50206f1fabb84cb2553532734b0957f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([22, 197, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_719add0b2938e656a9e57ed85bb60b53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e695181b040d543a64b309f7e853d595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d3d3d2f3788a455e4d657f47c6fd5ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e1453bee953b837804cc3f432461b0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 64, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_204c3f563712120081b552e20d68f275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e695181b040d543a64b309f7e853d595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d9e6d11ce7f8648e65a03a4bf434aead(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([11, 704], dtype='float32', min=0, max=0.5),
            paddle.uniform([704, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d23c9e3dce9ce701a518e10aff6fa16f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4b93157afb19598e9739029a9a9884c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 640], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_883d3fa270bffff9b9c5dd37cbc658d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 64, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_423f99cdb2331ff98cb250d16399b004(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d23c9e3dce9ce701a518e10aff6fa16f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a19d4b905ec237f787a2b7592cbfc7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
            paddle.uniform([1248, 312], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0f356681c39916c761b4be7cad9e5bcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
            paddle.uniform([312, 1248], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_40798a876ce5ab9cb134850e63828a47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f07baf684d672c5a19c3b3bf59f13ad9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_61cc39b1eb1f663c86be6b7f8f56a9f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d16e2b9793fbfabec71884b731cf21dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ac012aa70554f87f668667aed57c644(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32ac395996971744aa1f9edf6f911eb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0b6cf5cea6d7b5dd7ae6caadb06a9201(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32ac395996971744aa1f9edf6f911eb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ea7caa079b71926569f717f876e18ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c1b886c63f8f10d39da3e0f4ca2958ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12, 64, 1025], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a1a3d9f71c0159a0eee0575a344b56b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_88989f0bc8c1fbc298cce65886316501(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0b6cf5cea6d7b5dd7ae6caadb06a9201(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91fdbeb226cbb0729db8320a23f698d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
            paddle.uniform([156, 39], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1745a1cecb9d00b5b5ec9d67fe4ea58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
            paddle.uniform([39, 156], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efafcd3c8ebca9d487e8f03b58bdbbf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6747c92bdcb46849faedaeeb529a9ae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c7609f5f3c0dbfb0d1cba5e61e9a71a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 64, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_be2b81ad3f784ec434fb45a4af0763c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efafcd3c8ebca9d487e8f03b58bdbbf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf85beab0df7ca8be73cd0e87ebe9317(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
            paddle.uniform([872, 218], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfd6c227bc326c9cfe66082617440bb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
            paddle.uniform([218, 872], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2c7e2d4a4abaf3c9c1262c7cadf5c014(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_719add0b2938e656a9e57ed85bb60b53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f470a46364c9842a9ff0637565751431
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_de92ed05a1ce16a52f35433d504082e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_49a9675bdf48b860841d4905f447b550(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff3ebc32c6b144d0e969b47d362ba308(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db6fd0c214e473fe3d35a6b7e98b8d4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c03afaff2ecb46405ff2d0d3f099d15a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7d289397ba34502fe26253d1b8270fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fa01a146bfc60ee9bdd207ca3b78fcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b4b282c38a3db4b7bb49f547d43834b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 25, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 37], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_19ba98466f8c59011d37e12df02ac79c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_012fbf50e091e40c2840f2bbaaafd4f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_97d4b012236fda01d979151e9aa4ae4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7ba88f1950b3f58c605f1b4afdc2616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.to_tensor([[7.380273342132568, 6.993586540222168, 7.374476909637451, 7.281966686248779, 7.989966869354248, 8.266085624694824, 7.489744186401367, 8.130762100219727, 7.13471794128418, 7.360182285308838, 8.598377227783203, 7.828280925750732, 7.041106700897217, 7.472853660583496, 7.848850727081299, 8.005196571350098, 8.92200756072998, 8.074055671691895, 7.2128825187683105, 7.177803993225098, 8.775609016418457, 8.413022994995117, 7.756340026855469, 8.30650520324707, 7.593331813812256, 7.764378547668457, 7.462083339691162, 8.445094108581543, 7.404086589813232, 7.836207389831543]], dtype='float32').reshape([1, 30]),
            paddle.uniform([30, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6da4683462ae225c0589ee7db973823(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_122286752744cbec362b29457f8a7f41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 640], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2772c0a3abff27a64cc40c1b2845df4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 64, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_746f234dd8717f50a37d9983020d4eb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6da4683462ae225c0589ee7db973823(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a5f548976a3844166b096a5124ba06bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_43681c74e2007d69696fa3697aae9c1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4925fb4ecc0d5ed44d48ae72c1a1f635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 32, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b468fb379def1fed6974e68fddb5ad48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a5f548976a3844166b096a5124ba06bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5684a5518f007c04bb8966cd363c8932(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83c9577951e418c1f18e3e1782797dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffc7be1808b06b37bdd8f818290be0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_149cc8cb6bb06fb12ba654d760dc6616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_442db7e771031cd7c61cabc2fe9401e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e615911c6c19fc2514536ce66f0adcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2b336589bb49dccb05b001789325172c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 64, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ca4efd92ac43c307486225742e908c82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_442db7e771031cd7c61cabc2fe9401e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a292a22fa8f555e2d38e85e33c368a0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86602fcbda32c3ce7e62c4f80c0c85a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c487478543298b3263113c654d090db3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f473e93aad6498276dbc9542fdfe8ccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c51653b028c96d893e1bf0e7c1daf5a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4e9565e6aa84dece5d43f5554e44dc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72104d040c386e139ad11d0e322ee685(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_51245619844e8f0eaea625e0c514e8aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c22ea8292ead3c36a4f80391e2edf547(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c1d0d9b110692e9e6968745649005b58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 32, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17b9b3455a90b8bd2bf25929a4a3f25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_51245619844e8f0eaea625e0c514e8aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_80b1d12f36b78bbebfd78550a8309525(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6fe48ec2f4ca2f9c40d6c4db0afe6e13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_62ed02704ebe6d7afa2934a301328638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 32, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa57ed93052efddbf3ba4109c0de5d67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_80b1d12f36b78bbebfd78550a8309525(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_197bdeace673caddca0c14b3a9480e8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6bd2d5438c09df32faf2add2979a3af0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([86, 3, 64, 197], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8b7745756ebf1ff6b307c929a88e65d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 197], dtype='float32', min=0, max=0.5),
            paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2428be3bed908552f2922017010e6678(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8080142e3c8757ea22def88be9063eca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7fcaef35ec832c79b4d9dc4db75143a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_447935a734a68b35424417d8d80297c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1e62d59a916235649f8614a438e4d30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8080142e3c8757ea22def88be9063eca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5edabc654b9d3ca135fbde2ee011e4a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_305b1851c98ccaad4ea210c1dfc8b3c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 8, 32, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5e907711b646a47022b4ddd7a09678b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1ae60eb5aceee9c7b993e7e340001e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_382b1dc94ee36532a8b48499a6850b45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_40d7104e65f654b13dad18a0b5a03b2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6, 64, 1174], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5204317ba5cd3eba0fe80f80de36729(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b80ba77afa4802c53199ed0cb828806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25570026a9ebcd89bd03f76054db8717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([11, 1280], dtype='float32', min=0, max=0.5),
            paddle.uniform([1280, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d9d315aa6fbea7289fe2541819c9fbe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([43, 704], dtype='float32', min=0, max=0.5),
            paddle.uniform([704, 1000], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1bf7d57d69219f50ac514e20aa3f18a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b03e5286ee77b601eae4552ab303b389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12, 64, 1174], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5f4fdabdcb44843c01c738ec3eb34768(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd870a56a32301af78ab32c13cea3da1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8e98f8277562ed7bcae0aa2f62f846b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c22ea8292ead3c36a4f80391e2edf547(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92fdc2a965312fe7ef4e5db18c442f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 64, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f0e192c1e3343d1f65f7d19ca58bf4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8e98f8277562ed7bcae0aa2f62f846b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf83d8ebc28d75868f985b1d1f0a8458(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
            paddle.uniform([624, 156], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4a962ffe3011e5d13b09c62c917edc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e848ee7f0d57fd9d5291fde89933141
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
            paddle.uniform([156, 624], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56863167d1f048ae12456452e8aa7048(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7993f10b50bf7732a73968a486e0d1e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 8, 32, 50], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1fcd1392fb07d7c7200b1528d5cb9017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d53e86599aad1261d0cd947cd92fa6cb
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_420f05c73aa488dcf529f20ac4bbf5d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31dcef1dda70206efa90c6c82061c0ad
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()