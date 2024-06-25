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



class PrimitiveOp_defe42a6906c6b7513f92de86978631f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6cd875e03cd7daf12a4084a1c207444c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c7d2521754dcab6e57d441cf53a6c32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 576, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8f8697189f9859721a8a7068ec7d8ab6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [4, 4], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_948e04f07f2eb8779feeb025bf57b0c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f8697189f9859721a8a7068ec7d8ab6
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05eff6479d288852676d3ce936ce951a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c47591626b6c293a2b63586654439c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([576, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_924cc96d3d51e9ebf0b38366cd3eaadd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4aab61e9523361c3d72e2da1849bdc52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff41168956ad907504cfd95d5c6c3c91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d1931eba524fdfe8a5d0bec6a7397f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8507f8f1d78a3a2781c4a20682f133ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f8697189f9859721a8a7068ec7d8ab6
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0006586172e2c177009d680e1e7c760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 258, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b54285e85a5ad2cfb2f2271470a7fcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98b7731f6613b2c4b1fe0029fed3815d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d90aa6547b853e2f95092b39a3dcb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58616a483ab1b78eab2cc5f884219904(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08ee61135cb4b83828ba1d43dc8d2c82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 40, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_860695c01e6885009e0c64097da8bfa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 23, 23], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9ae74335c7c4eb02fd52f262251dad4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 46, 46], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e2751be06e188d4b32f3d147099258a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f6a427569aa41a234ef837e8e34203c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab18a224b51870c5f1b985b978c49cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31242a3fb990bc9aaa79a0d82ed50d70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9a017c6369133158ce1f64c91d9a379a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b3e2479d35a152433be9c904b1b6ff38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([16, 64, 32, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0469cb60fe1aa5381faf4b9168488b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 84, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 240, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efbdca9d353a0dc70a999c0eab6c6d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a70f678450ee47c9942ce2efb794697b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58d94e3d090f30e75dd90c1b1be35f1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c599c6f2443110bf94d4e4c5daf6fc49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.878262519836426]], [[7.868306636810303]], [[7.780619144439697]], [[7.209699630737305]], [[7.433918476104736]], [[7.6283040046691895]], [[7.112617492675781]], [[7.02634334564209]], [[7.322437286376953]], [[8.220512390136719]], [[7.710140228271484]], [[7.590466499328613]], [[7.6705546379089355]], [[8.223761558532715]], [[7.7061309814453125]], [[7.289975643157959]], [[7.402900218963623]], [[8.340198516845703]], [[7.1758904457092285]], [[7.918051242828369]], [[7.745652675628662]], [[8.17780876159668]], [[9.052896499633789]], [[7.247939586639404]], [[7.934576034545898]], [[7.615261077880859]], [[7.53016471862793]], [[7.217159748077393]], [[8.381265640258789]], [[7.74073600769043]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0035269ee030e56b058ca4cd9be2475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8e5e1bf195abcdee80034eabdb23c9ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d181ff9fce11c8e513a9aaa565c4248b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3c0dd5df0a0d0e63b6034ee061e51a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3467881679534912]], [[0.2242114096879959]], [[0.17263250052928925]], [[0.4016491174697876]], [[0.4109598696231842]], [[0.3546530604362488]], [[0.008467553183436394]], [[0.2828882336616516]], [[0.33728498220443726]], [[0.42666739225387573]], [[0.35843798518180847]], [[0.267415314912796]], [[0.04018489271402359]], [[0.29764625430107117]], [[0.013849708251655102]], [[0.1807769536972046]], [[0.0014414142351597548]], [[0.1628553718328476]], [[0.28109055757522583]]]], dtype='float32').reshape([1, 19, 1, 1]),
        ]


class TestPrimitiveOp_408d50dd31f8f64474a164935618d860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_408d50dd31f8f64474a164935618d860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_408d50dd31f8f64474a164935618d860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_408d50dd31f8f64474a164935618d860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_408d50dd31f8f64474a164935618d860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_408d50dd31f8f64474a164935618d860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_408d50dd31f8f64474a164935618d860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_408d50dd31f8f64474a164935618d860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b35bfde887e3aca536ae0127ec3ddcc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8e4e069c22a5b517f2e01a2c04b2f648(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c96f905d1208e391dc36611aabce14db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c96f905d1208e391dc36611aabce14db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c96f905d1208e391dc36611aabce14db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c96f905d1208e391dc36611aabce14db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c96f905d1208e391dc36611aabce14db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c96f905d1208e391dc36611aabce14db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c96f905d1208e391dc36611aabce14db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c96f905d1208e391dc36611aabce14db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_59992fb8cfc8f28689731f828fa363a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ef898eb6cd1cc866a265eed5c2ce2c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_912018ef568afbe72e6d0cf02f65e70e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_912018ef568afbe72e6d0cf02f65e70e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_912018ef568afbe72e6d0cf02f65e70e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_912018ef568afbe72e6d0cf02f65e70e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_912018ef568afbe72e6d0cf02f65e70e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_912018ef568afbe72e6d0cf02f65e70e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_912018ef568afbe72e6d0cf02f65e70e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_912018ef568afbe72e6d0cf02f65e70e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_63b269f6089720526c16ebccb4080887(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e936da3ecd1ebc22579ac7ce20142e4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab0b67158fca13f6f0a6fd0112dc8d92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab0b67158fca13f6f0a6fd0112dc8d92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab0b67158fca13f6f0a6fd0112dc8d92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab0b67158fca13f6f0a6fd0112dc8d92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab0b67158fca13f6f0a6fd0112dc8d92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab0b67158fca13f6f0a6fd0112dc8d92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab0b67158fca13f6f0a6fd0112dc8d92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab0b67158fca13f6f0a6fd0112dc8d92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9bcf8383930e10bb2eb1047fe542641c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_706ebea83ce668412a82ec0be6b01c22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcc7a1142bdddc082d6fb33407b7e63e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcc7a1142bdddc082d6fb33407b7e63e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcc7a1142bdddc082d6fb33407b7e63e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcc7a1142bdddc082d6fb33407b7e63e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcc7a1142bdddc082d6fb33407b7e63e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcc7a1142bdddc082d6fb33407b7e63e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcc7a1142bdddc082d6fb33407b7e63e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcc7a1142bdddc082d6fb33407b7e63e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57637b20fdf49ff4860e7e3bc37c4309(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b61d90edb0f7a1939bbb59b4713917c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7c1abd8709fe453c3277670e0ef0c00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 768, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_769136e44d52b7559b7d4523260d26e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b46b77560e9ded1817d04d1006f23861(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 768, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_62be4ea7fae25aabba8cca819275b762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([960, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d25f0f58b0eeb98b5c72f4087adb3704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 576, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d123c947d0664e710f7429a66165d4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 21, 21], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e08326a38b95939f2e4069ddad80961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 42, 42], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a09e90b6fb5875e9b26b0e7650233035(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 84, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_934283be20c71784395f5492e9ae41f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_59fa8ce8a56b37cc9092edd7415f835e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e1cfd9d3c3afbfb5baaca59056408b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e6ee55823199c82b05551890046e13d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([72, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd02bf9b6bffa78d4a93a3921fa8c692(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58d94e3d090f30e75dd90c1b1be35f1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83b464eb9745c0f82c4b5a76f9eecf16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.262385368347168]], [[7.8208513259887695]], [[8.153039932250977]], [[7.087778568267822]], [[7.932410717010498]], [[8.081872940063477]], [[7.86402702331543]], [[7.80531644821167]], [[7.08975887298584]], [[8.170979499816895]], [[7.904594898223877]], [[9.098105430603027]], [[8.166693687438965]], [[7.506779193878174]], [[8.451950073242188]], [[8.012106895446777]], [[8.47104263305664]], [[6.885653972625732]], [[6.992383003234863]], [[7.868102073669434]], [[8.112833976745605]], [[8.527132987976074]], [[7.738118648529053]], [[7.577332019805908]], [[8.22385025024414]], [[8.072021484375]], [[8.153665542602539]], [[6.797152996063232]], [[8.04731273651123]], [[8.220541954040527]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70371049c637bde51c9171828a81780c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac3e40c79d26179f261fe6cb1c297181(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9730641a85503d26bd92cd641010ea1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b3712b399ad7211974fb404a562c75d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b839350050a1713c3364566c290ba63b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 6, 6], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7a0ac3f46acb57efe2afdfb5c4855b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0767248198390007]], [[0.2148003876209259]], [[0.4181142747402191]], [[0.3729199469089508]], [[0.051715679466724396]], [[0.20637866854667664]], [[0.47527655959129333]], [[0.12755189836025238]], [[0.24502064287662506]], [[0.35078296065330505]], [[0.1111145168542862]], [[0.006618492305278778]], [[0.2509807348251343]], [[0.15735742449760437]], [[0.055007535964250565]], [[0.13243414461612701]], [[0.4190234839916229]], [[0.21615876257419586]], [[0.4291207194328308]], [[0.12087158858776093]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([5, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_937b6ce8f31ac5318834724a5452b9c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3657139539718628]], [[1.2159758806228638]], [[1.2202329635620117]], [[1.270909309387207]], [[1.1494554281234741]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.uniform([20, 5, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a85896acf06ed79b107f6fbed029208a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70d332410ecb87501e52f064901d68ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.6581876277923584]], [[2.988999843597412]], [[2.862517833709717]], [[2.8966684341430664]], [[2.84088134765625]], [[2.676283121109009]], [[2.9992473125457764]], [[2.9194250106811523]], [[2.943911075592041]], [[2.860955238342285]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.uniform([40, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9abb18844e20427c2b19fd9f99857ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7868a303fac26c9f04f1a3b77fc27739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9e2eab064255804d0a8334b99e2ead2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([960, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5586349df1449ff47eeb1060d89834fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aaebf188b45bdc9a461234b108c6212a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.257889747619629]], [[6.18993616104126]], [[6.881523132324219]], [[5.47817850112915]], [[5.700344085693359]], [[6.758537769317627]], [[6.853168487548828]], [[6.237391948699951]], [[6.216843605041504]], [[6.320457458496094]], [[6.477090835571289]], [[6.859305381774902]], [[6.186976909637451]], [[6.206911563873291]], [[6.260547161102295]], [[5.5532612800598145]], [[5.304288387298584]], [[6.289677619934082]], [[6.5896077156066895]], [[6.015058517456055]], [[7.109218597412109]], [[6.313172340393066]], [[6.01460075378418]], [[5.406548976898193]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55de2957a6bc36278013fd3a1f0a1aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_725db1397e23a27137f982417abdb9fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82ad1b5c9d4863b1f7b444d9d601ece2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([128, 128, 16, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff0449079dffb8d2320a94afdfc6b818(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f6e85223ca1a8da1f041a5d726048efe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dbbce71d20fc31e30c3c9ffda7d06e1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb1f61cbb788f8970388395fe3b9e4df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02850010e9988e9d71993bb2bdf38948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6838fb5260105e08f6049e3815e80ee3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7e308ee428732cfc424d68bf94de633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98f694a7afffc8b7b82ee32dedd06592(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8991941ab6425a50abc5bf3f23bc1173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc531b285c2283c40ca50e2d449b1e8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_725c5dc03f3655fff8db12430eae8451(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6088dadc66dbfb9fe219d3013180e1cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a8180ba8440fc6853904141f2af6d6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a1f5796c21dd0d3659a0b17b120ff55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07f60f85f17b4f7bf52b13532abb09ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b54285e85a5ad2cfb2f2271470a7fcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98b7731f6613b2c4b1fe0029fed3815d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ce04d639b8a8b6facf7c5f513cdb239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3fec2b66b6cccb14ae313bd2eca8f3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.22940731048584]], [[5.363189220428467]], [[5.409626483917236]], [[4.828910827636719]], [[5.264454364776611]], [[4.615478515625]], [[4.274387359619141]], [[5.118419647216797]], [[4.849647045135498]], [[5.2210564613342285]], [[4.860780715942383]], [[5.749593734741211]], [[5.439781665802002]], [[4.932239055633545]], [[5.167477130889893]], [[5.709780216217041]], [[5.35679817199707]], [[4.660642623901367]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7868a303fac26c9f04f1a3b77fc27739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9e2eab064255804d0a8334b99e2ead2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([960, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4067c8aaa7f6c1b6543170e98a4c1137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb1f61cbb788f8970388395fe3b9e4df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02850010e9988e9d71993bb2bdf38948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20538c2d00aba91ac32b0653f200bda7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e8c2cc5b68e73b4b255c87ccbd8c009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1152, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d91ed53660769be675e12511cbe0a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9abb18844e20427c2b19fd9f99857ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f630acad245ead1aeccab9895f46e3cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f7fe5de2596e8426185e0786e198789(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e08d6f84705090eafd34656c57ee2eb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38d5d920e72387e82186fbcac714977e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7681b69b63f2a1606f9d249cef01a3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2dbcc4718dd499c44d8f84df59ae4c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfc1eb7b5132e4da8d731b025b56af24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_906fb922b198b206659f95c05e6259c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_77f91673465af4aa0cbec40d9db37c07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70371049c637bde51c9171828a81780c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dee97751db29f0c4d0677d5364331d1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ff6d94a5f4992d259f31063ce7101b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6eb7403ed49c66c791f52abefad65ba6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([76, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5586349df1449ff47eeb1060d89834fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_369f577bd79aef983de055140db9deae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.02589750289917]], [[6.214494228363037]], [[6.199380874633789]], [[6.936142921447754]], [[5.8849263191223145]], [[7.262293815612793]], [[6.2682576179504395]], [[6.8438286781311035]], [[6.099343299865723]], [[6.04555082321167]], [[6.758257865905762]], [[7.5189290046691895]], [[6.542966842651367]], [[6.613718032836914]], [[5.846857070922852]], [[5.824804306030273]], [[6.4262542724609375]], [[6.430300235748291]], [[6.610617637634277]], [[6.656274795532227]], [[6.237132549285889]], [[5.95920991897583]], [[6.213052749633789]], [[6.1755523681640625]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d91ed53660769be675e12511cbe0a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c76273fed3d5bdf4790a2967ef14fddb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05872a0361108a4e94fa573ac9688455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32b9cc6bec8ead9131aee88fea9cd86a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d91ed53660769be675e12511cbe0a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fdc7a4baeb00a0ae0dfe6ce4d6d982a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_30385130000e8932ba1756a0d63e35a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac35658e9a4714e75412a3ec7ec46824(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.2939881682395935]], [[0.27856212854385376]], [[0.38506150245666504]], [[0.18114490807056427]], [[0.15810593962669373]], [[0.04467180743813515]], [[0.16293379664421082]], [[0.22157399356365204]], [[0.3598041832447052]], [[0.2736942172050476]], [[0.4819704294204712]], [[0.16151084005832672]], [[0.4532450735569]], [[0.2335914671421051]], [[0.3913910388946533]], [[0.22903932631015778]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.uniform([4, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4e111b95b1dd5184e115105c78f2a022(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.498032808303833]], [[1.1699885129928589]], [[1.4005337953567505]], [[1.0293478965759277]]]], dtype='float32').reshape([1, 4, 1, 1]),
            paddle.uniform([16, 4, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b961f9a512dd2941c299f59f8db80d66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([78, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_288f002b2607fb4d9fd9afe2f0b17ebe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([78, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d562f211af59da5d55413f4bcbf2e4c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([78, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d91ed53660769be675e12511cbe0a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3315a58935ea16afcf9c2a55cc1bc35b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ebae46b18a8ea71b5170b76e7b60fb70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.946103572845459]], [[3.4548959732055664]], [[3.2269513607025146]], [[3.368286609649658]], [[2.8261561393737793]], [[2.9311068058013916]], [[2.6655144691467285]], [[2.6224160194396973]], [[3.169365167617798]], [[2.677142858505249]], [[3.360309600830078]]]], dtype='float32').reshape([1, 11, 1, 1]),
            paddle.uniform([44, 11, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4aab61e9523361c3d72e2da1849bdc52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff41168956ad907504cfd95d5c6c3c91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9abb18844e20427c2b19fd9f99857ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e2dfd4e9ed8d4a9064081895ff97b201(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a084eb5315f5bfa8bc246dd52b5c1313(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c8a80c0140200f02ca9d4e5c4216351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7868a303fac26c9f04f1a3b77fc27739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9e2eab064255804d0a8334b99e2ead2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([960, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2beb2c0ded346856cc0bd15a976ec10e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a9b0ed433bceca2b272fcdab7cece3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_576da73f0f28d153d4f7fa1f448c22cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fc79abbcc1b6a1d36db22a299b3a86e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 288, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36aeaa139b07137d2b93e7fcfcb304b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58d94e3d090f30e75dd90c1b1be35f1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ed23c6e3d4ccf21b33bff60b6c63fc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.565597534179688]], [[8.85666561126709]], [[7.50863790512085]], [[8.139659881591797]], [[7.596431255340576]], [[7.8915324211120605]], [[8.013626098632812]], [[8.710958480834961]], [[8.468538284301758]], [[7.645221710205078]], [[9.079865455627441]], [[8.134474754333496]], [[9.158711433410645]], [[8.963510513305664]], [[7.479450702667236]], [[8.753923416137695]], [[7.598970413208008]], [[7.580456256866455]], [[8.508934020996094]], [[8.746679306030273]], [[8.63934326171875]], [[7.524258613586426]], [[7.697452068328857]], [[8.258296012878418]], [[8.279424667358398]], [[7.730199337005615]], [[8.538504600524902]], [[8.165992736816406]], [[8.467819213867188]], [[7.427496910095215]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efbdca9d353a0dc70a999c0eab6c6d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a70f678450ee47c9942ce2efb794697b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_da37eea445df6759ec5737fbd9b8df56(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [8, 8], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_67cbfa62aeaef830e5debff3db2fc78b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da37eea445df6759ec5737fbd9b8df56
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a5ec05763c9bbfbb53eb837ad87d0e6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b504eed7e71da1d1326a8bac662b450d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([34, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f477188686ab12ebda099afe655f96cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 270, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 270, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5f871ab12494b090ef8824bed88c8e89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_408d50dd31f8f64474a164935618d860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c96f905d1208e391dc36611aabce14db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_912018ef568afbe72e6d0cf02f65e70e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab0b67158fca13f6f0a6fd0112dc8d92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb529159b082e9140aec0770ecb13455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b857a3eca6a669a3c962bcf7ef82313(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65182bbf25abc93fd68d32dc2d87ae52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f536c725e61775d44a36ea4d21368af5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7c6d452534dbacb41e69ffc852fd631(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56c309dd56a10e70b1ea45b01d178867(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.462280750274658]], [[4.655839920043945]], [[4.034661769866943]], [[4.623201370239258]], [[4.026586532592773]], [[4.085882663726807]], [[4.6295390129089355]], [[4.664148807525635]], [[4.463987350463867]], [[4.678213119506836]], [[4.240274429321289]], [[4.858091354370117]], [[5.04475212097168]], [[3.7081854343414307]], [[3.8651325702667236]], [[4.311214447021484]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bcb6cb6581d7dd710d192281732d6a2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_63e61176fed0e7ec16663f0df620e456(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([76, 768, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e55a0be239450d01952f536040dd6111(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 240, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4def1f0a4e304f7434e79c034b1713c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7fc9494f6d3b9e9336da05d01160fb69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 7581, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.25520068407058716]], [[0.4102713465690613]], [[0.17479830980300903]], [[0.43605971336364746]], [[0.470920205116272]], [[0.26021814346313477]], [[0.039046768099069595]], [[0.1777527779340744]], [[0.4801710546016693]], [[0.28876662254333496]], [[0.3997771143913269]], [[0.13589894771575928]], [[0.26281604170799255]], [[0.08050616830587387]], [[0.12596160173416138]], [[0.011456800624728203]], [[0.049931347370147705]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_dff2a26c984aa23f4012c24926de8649(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff4b02abde9e66ccaf2108130c87798e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6bf803cff5f9b577d45b9ba81e0215b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d1931eba524fdfe8a5d0bec6a7397f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_40f6d4e31b0bb97bccb3987299f8923c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a692a546e181bcaa83a44752b69f906b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a692a546e181bcaa83a44752b69f906b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c884a8bb3f1c521bbf0a1802b70f1ae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0c6c1cb5cd974e8053cd45a6f238a46f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_26af07cb481c6e341be76d946499abdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6b443b1bcf440c8fb1fdd91b76cc98b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([8, 128, 16, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0965ca022b686e8a31b09ca187b5ccca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 4725, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3848187029361725]], [[0.44963836669921875]], [[0.21190716326236725]], [[0.19986607134342194]], [[0.16318359971046448]], [[0.04263736307621002]], [[0.1391931176185608]], [[0.12741266191005707]], [[0.07150720059871674]], [[0.40763071179389954]], [[0.3430783748626709]], [[0.45125240087509155]], [[0.2716154456138611]], [[0.1379556655883789]], [[0.4898723363876343]], [[0.4039318859577179]], [[0.4313584864139557]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_9eeb63ad8df5e01c15dffd84213464b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a980f0fcdfb3482878e1a1bedde1631d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dbbd47b51d727fde584b9a7dc82f933f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ff70bd4082f91fe4bc4d370d37c57af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd9d1808d0cd8c812298a1ec888df2ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2fda51660b4d1a7709076f582bd4d874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([8, 64, 16, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2408849f04671cf48da711c84126dbfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 10, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d74d4058d79a5f5e3d292c624f58dbe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 20, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cdf7dee9bd8ccb3f685b94876d294ba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b3d3a8eb9c78da2655795ed8b35955e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b8cb5998dbb19919e9ad40492041efe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90dc349e31709f04b1b0cd80c53fbb6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efbdca9d353a0dc70a999c0eab6c6d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a70f678450ee47c9942ce2efb794697b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae7477cdc3fead64d528506b0e217cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([3, 256, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b54285e85a5ad2cfb2f2271470a7fcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98b7731f6613b2c4b1fe0029fed3815d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff0449079dffb8d2320a94afdfc6b818(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_026c227a4d3993b6568f6508ca648ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ef3fa52ff5e450d562c3a76116b2a67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7433316928817df0f664eb874cdb1eac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af54c607cda46313201fb1c7b04492eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([64, 32, 64, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 32, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a9b0ed433bceca2b272fcdab7cece3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_576da73f0f28d153d4f7fa1f448c22cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58d94e3d090f30e75dd90c1b1be35f1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ecddf0d664de3f158ddc60766425ed9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.979545593261719]], [[7.502386569976807]], [[7.407017707824707]], [[7.328598976135254]], [[7.504692554473877]], [[7.502708435058594]], [[7.600711822509766]], [[7.319614887237549]], [[7.540493488311768]], [[7.206352233886719]], [[7.858943462371826]], [[7.72793436050415]], [[8.392699241638184]], [[7.311511993408203]], [[7.200669765472412]], [[7.192445278167725]], [[6.876770496368408]], [[6.432420253753662]], [[7.926684856414795]], [[8.070440292358398]], [[6.884799480438232]], [[6.536991119384766]], [[7.266120910644531]], [[8.063619613647461]], [[7.6492228507995605]], [[7.3653388023376465]], [[7.466222763061523]], [[7.257701396942139]], [[7.193363189697266]], [[8.130621910095215]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbe2ef84f4e7fb03c2905a558a74fc8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cbacdf74fbf18665bb80e8d36bd3964(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_544dd291627d3a5f491850d977d08638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dcb174e9b7b17c2aa6dcf42512fd4793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([25, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2819071ee4f9c7f1188bf80ae73ac6c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.4261298179626465]], [[5.420971870422363]], [[5.311474323272705]], [[5.697546005249023]], [[5.705629825592041]], [[5.828099727630615]], [[5.449417591094971]], [[5.936191558837891]], [[5.457039833068848]], [[5.1325531005859375]], [[5.356573581695557]], [[6.186561584472656]], [[5.268749237060547]], [[6.0112738609313965]], [[5.64337158203125]], [[5.816868305206299]], [[6.055903434753418]], [[6.368608474731445]], [[5.593043804168701]], [[5.533454418182373]], [[6.144598484039307]], [[5.9172139167785645]], [[5.942316055297852]], [[5.2752885818481445]], [[5.371233940124512]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.uniform([100, 25, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2286d9bb282e891f4422c57e15de4f19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f8697189f9859721a8a7068ec7d8ab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_145103abc854a677d175f0ad1067feeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 258, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7d28d1e6314b080ab27a18e1bba8225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67cbfa62aeaef830e5debff3db2fc78b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da37eea445df6759ec5737fbd9b8df56
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9acfe23b9875ce9dc0be43d124fc2e6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 51, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 51, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_360001745075f2065cd1db520da36573(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([16, 64, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_860cce80152172b2f9ad1db754826446(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 2048, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd659a345d03fc16222f4aef8d423e26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 1024, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab5b9416ccc67c291693df718ba3bd39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 512, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_470f8c5be3498bcc636fb9d6b8699101(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [4, 4], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_715dc372cc27e97f47de7df58b161a50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e344e08ad43c8af1c3e441d1a8ac388(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0778432927b455a614b15c4fe4cfb00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a4a160e654b285902279117577dd7c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d04e3f32d2a97da6c8c2108fc70bf412(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e6f0fa104c0d33b627eef570a3587f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_804f1d13b4a55cbec4804b6805c12fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a98ece7ff9e655a1f14a97513972fd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d9a68c6c3e6e394e5639daa6477e2ece(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86309eeae89c42d3c25f138017a83913(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_93e712738448bc3a93e43f4105797af4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([2, 64, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7868a303fac26c9f04f1a3b77fc27739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9e2eab064255804d0a8334b99e2ead2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([960, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76c9537bf978bb9228b0640ddaf7f45c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f27db2ac44a701de9e2a2c718b183b04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 8400, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.26861557364463806]], [[0.35001933574676514]], [[0.3140535354614258]], [[0.046488918364048004]], [[0.2541579604148865]], [[0.05171840637922287]], [[0.1804732382297516]], [[0.29430875182151794]], [[0.49036940932273865]], [[0.008544296957552433]], [[0.4117666482925415]], [[0.022563835605978966]], [[0.19516311585903168]], [[0.27179163694381714]], [[0.11322573572397232]], [[0.38302916288375854]], [[0.2439667135477066]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_517699b76629a2eb3d40d149d5292ca6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8e47170451f01461144cd78d2f059cfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 38, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70182d86b234f805305129a4f2e2bc2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 76, 136], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed5a3fa2b42f13617d01d0b4e9068796(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8cf027b7708426bae74ae211adda62b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bce47cfc49f79003df3e9f504b61064e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efbdca9d353a0dc70a999c0eab6c6d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a70f678450ee47c9942ce2efb794697b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e914a5c916d91dfdcfb1e7065f0727ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3b9261a2bf6362e2fa881a1a7878c5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31146a2bd8bbe0b6f79553b97faafa0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_968f21388ac6e31b64882cdea8105252(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_63d6753db89ba7dcfbcc65866565cdfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([196, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9abb18844e20427c2b19fd9f99857ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83f4289d5cd5d97657290345af13c52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d9da517f8efc16f3d2fb2af0b496a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b44c008246fd9083b5121fc7f7e59201(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0863b19feb690bf3aedb7844d2fae799(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_533c7db79bac707632584d01abe44515(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.819231986999512]], [[4.967597007751465]], [[5.137096881866455]], [[4.944881439208984]], [[5.179754257202148]], [[4.973467826843262]], [[4.452411651611328]], [[4.99772310256958]], [[5.283167362213135]], [[4.811030387878418]], [[4.926609516143799]], [[4.973180770874023]], [[4.5081257820129395]], [[4.762726306915283]], [[5.076358318328857]], [[5.03634786605835]], [[5.133474349975586]], [[4.7684645652771]], [[4.720879554748535]], [[4.631831169128418]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5080345e201be840a96e685ca90506f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6044a793d55ffd9762ca05bedc9b3397(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([8, 64, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efbdca9d353a0dc70a999c0eab6c6d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a70f678450ee47c9942ce2efb794697b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cbacdf74fbf18665bb80e8d36bd3964(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_544dd291627d3a5f491850d977d08638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_68fa6b43f9069dd176823d5f852d6cd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_46fc48750dd7e1dd961ffc2b75e0a393(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.12479068338871002]], [[0.2855343520641327]], [[0.16866059601306915]], [[0.34741586446762085]], [[0.36133480072021484]], [[0.2113163024187088]], [[0.12796184420585632]], [[0.26377958059310913]], [[0.2887001633644104]], [[0.09250585734844208]], [[0.38960957527160645]], [[0.05935746803879738]], [[0.30627572536468506]], [[0.19060786068439484]], [[0.3041316866874695]], [[0.4954233467578888]], [[0.17917902767658234]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_5ff6d94a5f4992d259f31063ce7101b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0b958a388203af543aca4c8c5e70f047(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [16, 16], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_836dc3c2e49f4f3c490971cb0eab9cb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b958a388203af543aca4c8c5e70f047
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 3, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a9b0ed433bceca2b272fcdab7cece3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_576da73f0f28d153d4f7fa1f448c22cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efbdca9d353a0dc70a999c0eab6c6d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a70f678450ee47c9942ce2efb794697b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3ffddf92c8dcb1a069e9488626436ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92a140ad8f44de16125455a8f0effb34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bea7c56e6730ef5f8cbf98b9f2ac631(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 76, 136], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_884dddde9e526c471aa7d9085d612230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f8697189f9859721a8a7068ec7d8ab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 800, 1216], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 3, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ce04d639b8a8b6facf7c5f513cdb239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dcd6afe021c2d80773ae99bf2bfc2c0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.8294167518615723]], [[4.928394794464111]], [[5.4254655838012695]], [[5.18821907043457]], [[4.075259685516357]], [[5.00288724899292]], [[5.440080642700195]], [[5.044766426086426]], [[4.88026237487793]], [[5.161048412322998]], [[4.695656776428223]], [[4.971778869628906]], [[4.89567232131958]], [[5.618208885192871]], [[5.1241254806518555]], [[5.2426371574401855]], [[4.9540934562683105]], [[4.764132499694824]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e2a42ade8d32bc00d6c7dad7f5cba0aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 176], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5ed08178149276287e43c1951d0a5c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3226fd19dd803f79b5f67056f525b34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c654353ace01c93de69b364cdd033d7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b814fb36d29bad9e55a2423343c7a30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 176], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7402be712461b36132ead4320e2b282b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90d451cd3abac3b451dd5fb53d5e70af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a5f528c142ad8fd04bc66a933d76a033(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd3b2644896316d77a7b7b7fb7646fda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2498c8e446e41095cbb0510599e04e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 4, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b8c0528fe5918095f674406e4797cce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d677e45c453e7447987249d0767ec07e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([576, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a35b2d2e96c31e8d083d05f7482c663(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a35b2d2e96c31e8d083d05f7482c663(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_904d1bf3a0b761939cc4b63df90feec8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d1931eba524fdfe8a5d0bec6a7397f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05eff6479d288852676d3ce936ce951a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c47591626b6c293a2b63586654439c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([576, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_471dbb453d23ca0e48b34e399cc29d57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e8440f85aec3debb2b7553553872d929(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ca3330dd90d1562746028b1b63b20c30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2499c605fa398addc13b1d7992e262a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3027c96c773fa4e834758d11579658f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9be2241de91654e6c2b3217ca240e431(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a89ab6434b6430d40f4895dc71877e3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da37eea445df6759ec5737fbd9b8df56
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d677e45c453e7447987249d0767ec07e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([576, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_683f48b958a8af53633150ac7be502e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9debd0650f99860823e9a2191ec8686a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 768, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edd0ba1e5283bab6daa9a8ed1dca91f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9abb18844e20427c2b19fd9f99857ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09902eb9feb6ad8bfb9c768015ee9b40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09902eb9feb6ad8bfb9c768015ee9b40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f89b88b29ad4a80db416a636b3201805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_62775ac4eec0ca55423d1d07ab4337a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1cd9b12cc5e9cff38b18479a262e8509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02c684bd5ceb0fbe809106de34b21b07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([12, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02c684bd5ceb0fbe809106de34b21b07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([12, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efbdca9d353a0dc70a999c0eab6c6d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a70f678450ee47c9942ce2efb794697b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3a928993a5974e515de6089e8153fc3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([43, 3, 224, 224], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 3, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01a1290fef916ca613184d8eb246a895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d5506f23c77dc2e7fad61ed9ebdb3ba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d4e3a603c7a38f6580817aad787112f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8f36e8e946d7c41095982f2efd57f350(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d5506f23c77dc2e7fad61ed9ebdb3ba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d4e3a603c7a38f6580817aad787112f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7edca1d6835f3bd819b74b4b8518384(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d09d189ad5cbfe626e4d8b8839936a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_71a741d1798b5228119e1e5eea828d93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2144508992c5d2977d75db89efed9368(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_318091c3b31a7af4a42a6b08447a3564(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d58c0fe271bbde13df1445b8ccfa2c52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85732aacde1bff1558b9a1109b3823e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f9352f1a680339e13c433d2ea050b6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_63804744ae7391a4205d6e3d2df6ad68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0ca7eeca75f9663895c26c49640375a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f9352f1a680339e13c433d2ea050b6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_63804744ae7391a4205d6e3d2df6ad68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4eba9b9c8527f33ca8bd8720c52396b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_be8f3bcea330edb198b92643821606cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02557bded2d8a5dd4d4db84f50d7b7fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e51acfb2006860c788a158845c33e070(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47ca701d9a1b4e23a5fc56c2357295ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_875408cd5b504ff035374f11481023c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cd1f16a7fea9c389ecaa1dd243024cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e2a2c346e2d45423427efc9cad5da97c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83f4289d5cd5d97657290345af13c52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d9da517f8efc16f3d2fb2af0b496a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ce04d639b8a8b6facf7c5f513cdb239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17404bf1de7aa3d0e3250a88cb03fb2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.579583644866943]], [[3.987138032913208]], [[4.566008567810059]], [[5.609007835388184]], [[4.824344158172607]], [[4.967480659484863]], [[4.597592353820801]], [[4.221213340759277]], [[4.373418807983398]], [[4.443808555603027]], [[3.9890925884246826]], [[4.554945468902588]], [[4.303409099578857]], [[5.069254398345947]], [[4.908219337463379]], [[4.3355712890625]], [[4.356297016143799]], [[4.193089008331299]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5586349df1449ff47eeb1060d89834fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1c9293720d078bc5d792d84f12bc685(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.418142318725586]], [[5.7315592765808105]], [[6.051327705383301]], [[6.1229119300842285]], [[5.347337245941162]], [[5.3859968185424805]], [[6.045626163482666]], [[5.939547061920166]], [[6.08483362197876]], [[6.583146572113037]], [[5.6943359375]], [[6.144874095916748]], [[6.776207447052002]], [[6.042275428771973]], [[5.6396894454956055]], [[5.307939052581787]], [[7.0514750480651855]], [[6.645849704742432]], [[5.006920337677002]], [[5.742197036743164]], [[6.737719535827637]], [[6.357386589050293]], [[6.788431167602539]], [[6.193100929260254]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ed1bc72ce00cbfec0223946272e6e8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 21, 21], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 960, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ddccfd2085a6d1e0ae82acf87702de2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dab59acbd982d7cce588c03eaacfc831(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b8770b6e5da34956a213918afae56241(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_094e0766d62467c5b67609aa986de094(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_665c5e007740e67b88159863271ecc48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc8d13372a624e4c6b8f706fe76bc42c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ae3a11f57a49a25c5ca60861caaf2c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8e2f3a7b80d326cc9e75b0e72af5a7c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0905fae6b94a7a30a09d8a7cc2150ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([150, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ce04d639b8a8b6facf7c5f513cdb239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b233ab4cdd3229f26fe753c67a7ce494(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.553216934204102]], [[5.243834972381592]], [[5.057967662811279]], [[4.7062458992004395]], [[5.247849941253662]], [[5.114867210388184]], [[4.990291118621826]], [[4.798624515533447]], [[5.134045600891113]], [[4.944972515106201]], [[4.478386402130127]], [[5.6592254638671875]], [[5.583248615264893]], [[5.141046047210693]], [[5.647624492645264]], [[4.361642360687256]], [[5.095796585083008]], [[4.453990936279297]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_924cc96d3d51e9ebf0b38366cd3eaadd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9ed78012698563052b38b69ac6638eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1de872b5203536c1201c1c4e6b8e47b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1de872b5203536c1201c1c4e6b8e47b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ff6d94a5f4992d259f31063ce7101b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e99972f2026c7deb7e79dd2d21e03293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a6213285df22ec11f0bff694328a8015(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b5270960752a045a223886347bb562c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_321a3055f2ebc02ac65286518497cc04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [4, 4], [3, 3], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ce8014bcf450ca34213a2ae28fe4b7ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_321a3055f2ebc02ac65286518497cc04
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 1024, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 3, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3380a616588f168cf5cd4bbed8962d70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd11886dc7838c17747e186cf5150161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9328f17553fc6303c8dad67928e3c455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2fac65463bbbad87fe5b9568afe087b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1568, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2111793615d476c00521fd06f48a752(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8bb8e1ba98c2edac002d03ecf227d4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0578cd6caaa482cdd7691223f305aae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([258, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e7c9f866b9e94d5b5b0243f56e9aecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a4c33f6c57c9734e07f0648ac07e432(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_310563d8a9769e73452da18a6f071597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_310563d8a9769e73452da18a6f071597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ce04d639b8a8b6facf7c5f513cdb239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2fd83e9511011711c5d98b8aef74d965(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.06638240814209]], [[5.281164646148682]], [[4.484489440917969]], [[4.670993328094482]], [[4.920041084289551]], [[4.694016933441162]], [[4.3606858253479]], [[4.4463958740234375]], [[5.016234874725342]], [[4.9394707679748535]], [[4.267251014709473]], [[5.512999534606934]], [[4.754570007324219]], [[4.967658996582031]], [[4.833072185516357]], [[5.408027648925781]], [[4.468790054321289]], [[5.006646633148193]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_214ea0dbd87bda619f00c9daf6af0478(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a6213285df22ec11f0bff694328a8015(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b5270960752a045a223886347bb562c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_87a96aa1829ec132f7ca4fbd3c9fe68f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [4, 4], [2, 2], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a2ed5510ec5d8a25a9a1b9aa07a4de6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87a96aa1829ec132f7ca4fbd3c9fe68f
    def get_inputs(self):
        return [
            paddle.uniform([6, 3, 384, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 3, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_108df1850a949c31340b3b3512c25393(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25c89a67b6630fcbf426f6ff8f37eb18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb1f61cbb788f8970388395fe3b9e4df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02850010e9988e9d71993bb2bdf38948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b00d7dd18b3a3738dc4ae86a9e987e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e551e5c8dba335bfaa54aa92f65d32e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d105b35aa85d56796d8e8aa8caaaaa88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d91ed53660769be675e12511cbe0a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d265d4c0ef29bee98394c549970479d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1da400a6a013df72bf9e15a0ffc78e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4681ac6c401d0ef6eb71b06ad802148d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([10, 3, 224, 224], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 3, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_62ff4661b4bb6d3ce5e159da8fb401ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_94986f808f41a806f48780ed697c977d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1ee2dd69f0a6fb0e77d385085519005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92290ac35ad6edda1176337ee1ae98dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_94986f808f41a806f48780ed697c977d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1ee2dd69f0a6fb0e77d385085519005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_709cd3ce47058c7e2d2a2bdab6eedb2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c572e81494a03e0de2e3bc01bc71b221(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b928f4d419327bb2c16d6550adab6f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a2fdf7f13deb9b3c347bfa06e01df82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_414d2bdcc5fe835fdc85bdeb49f5f49b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d059a491147caa45ee6c33a4bde0cb77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b98dc777073ca6da31434dcc052365d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cb8b1ae2344d06ab61f068c5b762c69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f652a0c5ff4a1ff9fab8d7c4ca1a43aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eb1b350e53f4326ce2683399c0972b8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cb8b1ae2344d06ab61f068c5b762c69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f652a0c5ff4a1ff9fab8d7c4ca1a43aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d977fb8c0baf810d15ff49dae3d677e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f58a101efdcd99084fd439c1ccc1dd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f6082e865934a6d97fc988a234be97ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb3ac8308d46dcee738d9b3242a3e1a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_94cc1fff91a34892a9fc38beb0d3fb03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5ee36821a0be1f1edc4e7bb26c57754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c342366fd67e5eeceb1d74a6731fa5e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ea0662aa2acd80b2e2ec18bf2bcb55f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9debd0650f99860823e9a2191ec8686a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 768, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce3318ea808029f56edac1ab79709fe9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f377986ce9aed23ad2a17f8d645b8a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3053f37d99cf571489761652a2ebd4ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([34, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d5fe686e3307ffe52ec6d77d3c20ee8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_49cb161d0b7bfaa8cb0fabb27b497ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f72919153772386b1981e3bb12c513d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ef3fa52ff5e450d562c3a76116b2a67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7433316928817df0f664eb874cdb1eac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b175310087d2e9aca4c6b884e2c22af3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2fed590449e3b0b6c77c2666cba6354(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df8f285bd77b69e6aa47a92018f5920b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa944ba3a6f0a6052fe9fa9df835057f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 1024, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad569fd23a6dbbcb19d0b3cebe78ba62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([126, 1024, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c00a5ae1dbaf1ef0e0ee56b9fef35918(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf2cafbaa1dd92139ba4391ac4f8c9a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([126, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0245995f92ceb9659c9e51fd06ec0004(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d990a916f5beace7f649f66339f3e7f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([126, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b6c4aaedd0adf2e2f6e5823e0242bc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8cf6152d3f4aa55cdefefa1c641432c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eba252b27d8ec58ba2d113692ab23c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf2e85d979c2a1238b2c5009546ca9fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_471dbb453d23ca0e48b34e399cc29d57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e8440f85aec3debb2b7553553872d929(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f759320d14f8d4e1eb677568511edd36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_414520b85ef1f38cd1dc365e56f317a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb3a0425c93e638010e2e825f9f11ad8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_787684a425bbbdc5298260e14f5a6fa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ade5a5fe3b9c61fa672bc88e9d447c8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8c7563edf4be58dbed904e7ba92e35a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a17d192ffcdf81ace250d40613fd910(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1152, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ff6d94a5f4992d259f31063ce7101b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25d6a8410155d00c950aa56bd8f31abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 4116, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.14103396236896515]], [[0.1087898537516594]], [[0.49010708928108215]], [[0.4572169780731201]], [[0.4538974463939667]], [[0.345737487077713]], [[0.025614283978939056]], [[0.07712259888648987]], [[0.29893752932548523]], [[0.3660225570201874]], [[0.22165629267692566]], [[0.4913865625858307]], [[0.4068002700805664]], [[0.4600292146205902]], [[0.4365499019622803]], [[0.2297605276107788]], [[0.369172066450119]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_3182fdb6ed96af81cb8319195389c1e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2154c67e3507cf3de40e98c9576cb8b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02f19997507d51a620d80a58c601fec9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2d94039c85ec465c362a5b76121fb3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b29297e18046f80a50897da31219c21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffbd2c29e5e868e732454978ecee77ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([17, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9abb18844e20427c2b19fd9f99857ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f958a22e01999a6a36ba7842d27169c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d1931eba524fdfe8a5d0bec6a7397f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3d8ae0039941b28182875298838fa71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a091cb92fc718ce7e89f1ff9c4974ef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6fa1db336b991e4da73814c8da9abe38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a86c343f0598b559276b35fc8e1f8974(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_394c2249f770584d88e204d4d2ec0bbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffc37705d099e6eb65f4f10f83224a8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5aeaea6d467e9980b248248fb484609(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ec98bb357939d088405f53cf0e40138(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0985ea0d0a23fe67d4b964c448518d63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8641e76c008f0407a29b0fe4f109d777(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03499311eeec111e4c07f3df06b3a44f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_428a9c17742b8be7719885e69d702f5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_151f10c79c99129c9f7929c042778bf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a960c3fc46d9891d30083f9ba23ef19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac7ae32dd1ec8ae5059b0a6121f9825c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 300, 300], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 3, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_118d9da99e4a339f6169b890eebd3b20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d4806aa20f707377c1308e9b3c4f9f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 150, 150], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fbd0bb6fa7663e771ac337ece5802262(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_42b435191aba5b8a73db6102dfa94680(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 75, 75], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56edf42705cdd8019fff79ef14bb0469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56edf42705cdd8019fff79ef14bb0469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bec56e898d7bea8144a3034d3e52fc89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d91b9a9ed5fadde412e4f78f21ae06eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d91b9a9ed5fadde412e4f78f21ae06eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e30440101b81a0148962e4721c92349f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e30440101b81a0148962e4721c92349f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e30440101b81a0148962e4721c92349f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a18651d813c0c9eff1612c9783af1e01(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [1, 1], [6, 6], 'EXPLICIT', [6, 6], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a118c6c069d64af6fee23bf758c00e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a18651d813c0c9eff1612c9783af1e01
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe49d78f837f2fcecbe7252c8831912f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3a8d16a18721ee2d381fae4cded6a4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5ad12ab954bde8ec61a8a381402fa85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f515e2cad3b092eb630a4494bc34e8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a5560ffdda8a0fd46120a89259982cf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a61adda77e65a70f0f72cda93b48c655(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d420c6408179b28bb8f35d3f37cd7f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e427492ebd504b87e6b3169dd3fe8ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8258f49dd2c964382618efdbb552da25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_540aa893b2f07af4bfe203659b055581(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efbdca9d353a0dc70a999c0eab6c6d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a70f678450ee47c9942ce2efb794697b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_036c5a8ae5702386a17a17eb8782a1f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db4509fc8a3137d2ddd315b3dba06b8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6dad58c6c726a6d21dd744071b3bf93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d5243d23d0dbbd37fc05134b83443cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3e0803f0455e4b2845c1fcc1daed77d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([392, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_501f22bebfdc57a772bbcc89509b400c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_88cb4492c59c7333e5db3c8c6b16741e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 270, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 270, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af1e73158f235f5cfc598d034bca74b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_094e0766d62467c5b67609aa986de094(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_665c5e007740e67b88159863271ecc48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc8d13372a624e4c6b8f706fe76bc42c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df26173be3b23153e26996144aecac5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_48fcf54a2c8092f09b40d57889e1d100(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 96, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5f57f853a32bd7d76445406f121d9121(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 6069, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.06769630312919617]], [[0.18325413763523102]], [[0.36699071526527405]], [[0.48212292790412903]], [[0.3831770718097687]], [[0.14303450286388397]], [[0.24963904917240143]], [[0.47121456265449524]], [[0.3864501118659973]], [[0.251947283744812]], [[0.3882531523704529]], [[0.13550661504268646]], [[0.42309945821762085]], [[0.2914310395717621]], [[0.47562479972839355]], [[0.4436182975769043]], [[0.301053524017334]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_849428ea6d20654c9d7cb253bf390ad2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eb0fd3f47bd71cac7cf92910e487b8a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_15030bf8ec87d738a375904ea443c4bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c43b732208539e8466140b1470aa964(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2ea70eb54ce28826f84ca1e1cdc716e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a53c175b7ad36d1190f319d56a0cf9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b321a6cb0761697b840fa2180bad349(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52503e79eed7121ba87cc4983a8e932a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d3c66ab9569d9490b35ef3fcfeb3066(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_61775a54e07260548c4be1c5b05b534d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a88e055e3343190c13adfb8b549796e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd7c5851b6a62b7071deb8522ec093e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03fd5ae5169cc22a6df49493e754970f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e07d0b4e11d0b9461630d30e199cb56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af784c121cd902ad1ede6bfed260b6fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_066d8c22c6633331952a69c513a47a91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a3b9b431c22940542e1cc2f140d7c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 9, 9], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_802a6e659ad4790d43b3d7870477938a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc34f3662f830d2f2dc1ebbac30fa601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([150, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0eaa37e5d395dda62f90402103de4242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([288, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_924cc96d3d51e9ebf0b38366cd3eaadd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9abb18844e20427c2b19fd9f99857ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_108df1850a949c31340b3b3512c25393(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f98c6297a2f037610eb9e644bb1c11f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_332114e7539885115afda5716958a7aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e7ce3b04495077bc5c643015adeb3ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 15, 27], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d38dd3d285b4895df47f30bb8edaae41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 30, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9bc2980c55df6726190f179a6979566f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 108], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ce04d639b8a8b6facf7c5f513cdb239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73e5b7d1aaaa5b13cfdd10f18d39a518(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.6244988441467285]], [[4.329029560089111]], [[4.995797157287598]], [[4.316667079925537]], [[4.923176288604736]], [[4.73710298538208]], [[5.304345607757568]], [[4.629157066345215]], [[4.705936431884766]], [[4.469086170196533]], [[4.268731594085693]], [[4.755530834197998]], [[4.751047134399414]], [[4.774821758270264]], [[5.1132731437683105]], [[4.930696487426758]], [[5.313457012176514]], [[5.625310897827148]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83f4289d5cd5d97657290345af13c52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d9da517f8efc16f3d2fb2af0b496a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6ddae923061303f885896b0528819ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_381109856e50ce190f346e0a37e58c93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac9141519c80be41e40fd962bdec6df0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03bdd818623d9c3c04c312d660fd63c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_930041cf69db0eb68add516aa3d99c62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_933757c1fd0a8b4d60131547efdeb459(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5f871ab12494b090ef8824bed88c8e89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_408d50dd31f8f64474a164935618d860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c96f905d1208e391dc36611aabce14db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_912018ef568afbe72e6d0cf02f65e70e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4aab61e9523361c3d72e2da1849bdc52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff41168956ad907504cfd95d5c6c3c91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9e9e834cc06e1f838c72fcc8af40629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 46, 46], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b4a61366380b3f90d58aab4a617311c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d70594a75016a091bddf0e1ec973842(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_51b8f03b0600ede77d25dbfaf2c471ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d51198f417b715876583fb22df9b53f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_170e1f11ec20bbf4156617a2b70940e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b461e174da9fe18b73ef37cf5865d6e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_942a73f0e73db7dbe4fbdfa2b0f300f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a218af7299ea3a6b19b60e08f931a1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ff6d94a5f4992d259f31063ce7101b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7c6d452534dbacb41e69ffc852fd631(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_37c3c08c042015d370ee0bdab95f009c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.71750020980835]], [[3.766853094100952]], [[4.769667625427246]], [[4.383036136627197]], [[4.173023223876953]], [[3.5847434997558594]], [[4.281980991363525]], [[4.766328811645508]], [[4.207974433898926]], [[4.099235534667969]], [[4.517120838165283]], [[3.9880950450897217]], [[4.008715629577637]], [[3.37216854095459]], [[4.127290725708008]], [[3.942164659500122]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_de9e29510268046f432b5a8790116a4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_121f08ad4219075573fdbafd05c2e6ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_121f08ad4219075573fdbafd05c2e6ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_121f08ad4219075573fdbafd05c2e6ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_121f08ad4219075573fdbafd05c2e6ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_49cb161d0b7bfaa8cb0fabb27b497ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f72919153772386b1981e3bb12c513d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25ddf1622edcc8381dfcba8b389f3639(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25ddf1622edcc8381dfcba8b389f3639(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ea411e6ad59c8d666d87c0f5945ad574(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_71bc747be4e00d1c60becbce48b4b4f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_802a6e659ad4790d43b3d7870477938a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a9b0ed433bceca2b272fcdab7cece3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_576da73f0f28d153d4f7fa1f448c22cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ce04d639b8a8b6facf7c5f513cdb239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_33d9dcd053c7785dcb7523af357d1798(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.8236403465271]], [[5.052951335906982]], [[4.7229461669921875]], [[5.1916890144348145]], [[5.195699691772461]], [[4.67725133895874]], [[4.74697732925415]], [[5.135109901428223]], [[5.14848518371582]], [[5.57559061050415]], [[4.84277868270874]], [[4.417875289916992]], [[5.214022159576416]], [[5.045286655426025]], [[4.8187713623046875]], [[5.191462993621826]], [[5.3586812019348145]], [[4.862154006958008]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce3318ea808029f56edac1ab79709fe9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f377986ce9aed23ad2a17f8d645b8a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_924cc96d3d51e9ebf0b38366cd3eaadd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ff6d94a5f4992d259f31063ce7101b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f6c60740e7650fac9336b9769ad284f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.26703914999961853]], [[0.4573577642440796]], [[0.4691845178604126]], [[0.19444303214550018]], [[0.2478325068950653]], [[0.41838937997817993]], [[0.32610562443733215]], [[0.3988673985004425]], [[0.48955947160720825]], [[0.42022302746772766]], [[0.38033583760261536]], [[0.3768266439437866]], [[0.3617059588432312]], [[0.2924998104572296]], [[0.29406145215034485]], [[0.28213030099868774]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.uniform([4, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0bffc416da3f18a6268e6a8f84d19dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.9138317108154297]], [[1.692276120185852]], [[1.8661717176437378]], [[1.6739617586135864]]]], dtype='float32').reshape([1, 4, 1, 1]),
            paddle.uniform([16, 4, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f76f74a74041056222910797eee08d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 6, 6], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d91ed53660769be675e12511cbe0a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f31805476a849ae9fa20d3ed2737367(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f8697189f9859721a8a7068ec7d8ab6
    def get_inputs(self):
        return [
            paddle.uniform([43, 3, 224, 224], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 3, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fbf50404ca9725971a2850c9805cec5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([11, 3, 224, 224], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 3, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ebf37e2f89f1d51d58d705dc30391a4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_016d625f3b995169ee7f4cd24b914a9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b2298c294d81778bd7bbd505f92aa84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92e0836af8e018a1ac2b151b80b220fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_016d625f3b995169ee7f4cd24b914a9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b2298c294d81778bd7bbd505f92aa84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34012fa2d2ec914cf312c56e539476b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed68a3b76b02d9fd8289c4291dfbe309(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52ef65dc69da3823b98d632d8c2010d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_353ba6792b2d9f271d6f43c0f8e0b107(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d24dc53e4e7bc7db37771160e1cb1664(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86c3b3817e0564e1f07e5349430cd454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dfdb503b824afca47bb5506f3236d95a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8671541a6b14d03cff12332a42e3ebc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87a499ba76595d7e7dbfb51961309787(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fadb854ca2b560c253a8dda31ae848df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8671541a6b14d03cff12332a42e3ebc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87a499ba76595d7e7dbfb51961309787(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d8dbab1c1ee792ae87854fcec28a2f17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_93afaa5543bd0f1750c9a01bf1c44c51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff2f6d59dec2d4d5238b0416c450634b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_126cb3935a8dca417805f1369b8a19da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bed25029523a2b604d47a9b551e8fc0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ada28fd918b6ad2fbddd6900d51b7542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0c1f0f9575d39cd1b8feb1dfeb92aca4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83f4289d5cd5d97657290345af13c52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d9da517f8efc16f3d2fb2af0b496a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e914a5c916d91dfdcfb1e7065f0727ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3b9261a2bf6362e2fa881a1a7878c5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a3b2d7fe79395d6fd7390e41fc929135(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.conv2d(input_0, input_1, [2, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2bbd290f75388b2ac7f5a132c7e969b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3b2d7fe79395d6fd7390e41fc929135
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 4, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_139af9c430428a7b3592b485ee97ded0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13e9c7f0af0230538f5cabe2ab5fa87e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3de1823c2b4fc7292bf24b963459a74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7bc6055b7fd6f43983647df3c915df82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe7cb3be5f3ae1e77940bb75fdb8cdf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_88b47185a28d012f22a25bbdc0f4de77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5aaca99ff26dc39c3995b8691dc7ae25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_722a8df4dd692ead69f0769588be009e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14e1f23dc9a92567955b007856da934d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2478ca844e701be7cde26629b57bde46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 960, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4e178700c0f7e141492f27226a7cfe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c61f959d672980441635ce76ee5a58b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9178288e0627ec249ddd290e584ecea2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_531ea53ae7b65c13c8ba1792a7f7c191(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([400, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5f448638cd6cbdb1705d5060a3115b1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 160, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_77c677907e1ae32024ba6ecbe96873f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa93a23afa580ce1a178c1ad5dca2abb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_46d4168ba98eceaeb5c14fd630bb559d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64934e69cebdbf5509612da8ddf47985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2bf064b21c5e32abdc41f81b2147813(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8991941ab6425a50abc5bf3f23bc1173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc531b285c2283c40ca50e2d449b1e8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_725c5dc03f3655fff8db12430eae8451(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1265438b1b405465903e7fce795fe286(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 136, 208], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2bbb675848323c31b9249af1ccbb64f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 68, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b205f4fb8a7cd7c9edc11f2225d2dc5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 34, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ff3921b817c30cf123fec81bf0c366a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 17, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cd7987785a8ddafe965c3f721d0d7be1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 136, 208], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e7a19761cdf637c6e7597f47cadff8b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 68, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0b170a075694fb0c1ba1b91ecb8a57f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 34, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28e536c7fc90621a5212b25d7b14dc38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 17, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cd542d6004e3130b4825dbb80d8c17a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c9ec1cec7d3df82d2e61074cc0e3f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b6792f8e37c162942514c323017c8cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_081d434a9157d48d6b0dcdaed80fcfae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 258, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_be3316bf8913ec8b70d2b823e869ec8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 144, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83f4289d5cd5d97657290345af13c52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d9da517f8efc16f3d2fb2af0b496a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7a8d79e28afd4b95b9203d4b642f0478(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([16, 32, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 32, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e27edbe81f2b5059bae2e0e7fa5a608c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3b2d7fe79395d6fd7390e41fc929135
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 8, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_948e04f07f2eb8779feeb025bf57b0c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f8697189f9859721a8a7068ec7d8ab6
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_62be4ea7fae25aabba8cca819275b762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([960, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a9b0ed433bceca2b272fcdab7cece3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_576da73f0f28d153d4f7fa1f448c22cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a6213285df22ec11f0bff694328a8015(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b5270960752a045a223886347bb562c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1da400a6a013df72bf9e15a0ffc78e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efbdca9d353a0dc70a999c0eab6c6d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a70f678450ee47c9942ce2efb794697b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0863b19feb690bf3aedb7844d2fae799(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5010798108c4f0f74261cc367f534277(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.320950031280518]], [[5.029046058654785]], [[5.069671154022217]], [[4.538184642791748]], [[4.629547595977783]], [[4.70802640914917]], [[4.715033531188965]], [[4.512894630432129]], [[4.810446739196777]], [[4.68913459777832]], [[5.208148002624512]], [[5.160460948944092]], [[5.343917369842529]], [[4.764742374420166]], [[5.029192924499512]], [[5.373773097991943]], [[5.740528106689453]], [[4.85421085357666]], [[4.823370933532715]], [[5.062239170074463]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04de6984e95cce902c0fa64dee1a53f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7e476089499b9ea97a7f6822aa343a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ff6d94a5f4992d259f31063ce7101b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc12ff66dfe0f7d876d18fc5d379288a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([76, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_816806a6ffbb4c9c0641660a92951e78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 15, 27], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01b5a72a97d33654cee86ab7beb1c753(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 30, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_88fabbb8317311768392465162a2052e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 60, 108], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f7b721473cafd0c119d539eab4abac7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ed311747efc0448d40a9c3c36400544(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a195168c95cb5044c2f1d46db7970785(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3e0803f0455e4b2845c1fcc1daed77d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([392, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e7c9f866b9e94d5b5b0243f56e9aecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a4c33f6c57c9734e07f0648ac07e432(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc5a4d89f99fc5d66e13857b732f72ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([12, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acc9ec911caae94fe2a5920936c205a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.601583242416382]], [[3.2522401809692383]], [[3.9426109790802]], [[2.8303322792053223]], [[3.6804349422454834]], [[3.1349878311157227]], [[3.369419813156128]], [[2.978642463684082]], [[3.2317445278167725]], [[3.1392486095428467]], [[3.2146658897399902]], [[3.248555898666382]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.uniform([48, 12, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0863b19feb690bf3aedb7844d2fae799(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f6fdc04a2814de2365f926e2589f1bcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.887347221374512]], [[5.173083782196045]], [[5.676054954528809]], [[5.187004566192627]], [[5.235259532928467]], [[5.570391654968262]], [[5.4706244468688965]], [[4.995048522949219]], [[5.354491710662842]], [[5.069897651672363]], [[5.129307746887207]], [[5.034744739532471]], [[5.100893020629883]], [[5.21192741394043]], [[4.591923713684082]], [[5.437050819396973]], [[4.8461527824401855]], [[4.92936372756958]], [[4.811412811279297]], [[4.633780002593994]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3315a58935ea16afcf9c2a55cc1bc35b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_585923b259383f6473f7e46992e9b18c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.239675998687744]], [[3.233325481414795]], [[3.1844775676727295]], [[3.2806575298309326]], [[3.5525128841400146]], [[3.763702392578125]], [[3.6452975273132324]], [[2.8868260383605957]], [[3.655923366546631]], [[3.5652997493743896]], [[3.570770263671875]]]], dtype='float32').reshape([1, 11, 1, 1]),
            paddle.uniform([44, 11, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a9b0ed433bceca2b272fcdab7cece3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_576da73f0f28d153d4f7fa1f448c22cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f7fe5de2596e8426185e0786e198789(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e08d6f84705090eafd34656c57ee2eb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38d5d920e72387e82186fbcac714977e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2dbcc4718dd499c44d8f84df59ae4c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfc1eb7b5132e4da8d731b025b56af24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_906fb922b198b206659f95c05e6259c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1400c28a16da92d4d52756081617d748(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0985ea0d0a23fe67d4b964c448518d63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f39f6ad284f06f6cfa67cb804086346(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9178288e0627ec249ddd290e584ecea2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_531ea53ae7b65c13c8ba1792a7f7c191(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([400, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6427fd658eca21a04e27948a4197b89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_321a3055f2ebc02ac65286518497cc04
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 3, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8e5c74b5c3b44c4b85718b4a7897b161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da37eea445df6759ec5737fbd9b8df56
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fbb4b99dcb5c8eb8e38c7174a013aabd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03e1305e9e6ba2d0d78d207c72ed413c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10bf81f821fb42f449de2151ff39f7be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_745d519ccd310a56de3a32865b4ec844(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3b2d7fe79395d6fd7390e41fc929135
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 8, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_00febfd0f86ef30c06a953c066f44ef5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_88d2d9f7bda56f0fb075a6b5bab71054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([14, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2dfabf695a62ced30578f92b3a50a44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.8458921909332275]], [[3.893167018890381]], [[4.0064873695373535]], [[4.077722549438477]], [[4.0647101402282715]], [[4.12076473236084]], [[4.0326828956604]], [[4.035554885864258]], [[3.636183977127075]], [[3.9118173122406006]], [[4.488284111022949]], [[3.9961748123168945]], [[3.807126760482788]], [[3.8151307106018066]]]], dtype='float32').reshape([1, 14, 1, 1]),
            paddle.uniform([56, 14, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_482cae2a25c740fbddf60051e8be4fb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6aea6813185f0ad660891bb41bade58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08cc910abb701d47c4e825dc4880ad82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d17521ed67dadbe366ba54f1ce346ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b661be20633c7ee7a3a08420a025adc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2bf064b21c5e32abdc41f81b2147813(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8991941ab6425a50abc5bf3f23bc1173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc531b285c2283c40ca50e2d449b1e8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_725c5dc03f3655fff8db12430eae8451(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbdd04f06ab8b9b413690f5241ac741c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 576, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0eb452e275f3e9a81859d291146e3022(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 384, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_579e039cd546693935d88c3d5f20b7bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_469dc3203a8abb4c760d9c43bb982894(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 23, 23], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 768, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34fe61b6773ab75ed0ac6d61ff395513(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 288, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1bb23aa66b242ebe6e562ba6e7ce87e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([8, 160, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 160, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17f0b70978afb18f58ebccbfed08d1bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06d965ad93e9b3d2a9855fbe4a946706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3aeea9582e11652731a4a9bb2edb281d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef9467135db67dbb5a9f36aa2c81c6a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 144, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08cc910abb701d47c4e825dc4880ad82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d17521ed67dadbe366ba54f1ce346ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b661be20633c7ee7a3a08420a025adc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8991941ab6425a50abc5bf3f23bc1173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc531b285c2283c40ca50e2d449b1e8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_725c5dc03f3655fff8db12430eae8451(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6088dadc66dbfb9fe219d3013180e1cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a8180ba8440fc6853904141f2af6d6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a89ab6434b6430d40f4895dc71877e3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da37eea445df6759ec5737fbd9b8df56
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cd875e03cd7daf12a4084a1c207444c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b175310087d2e9aca4c6b884e2c22af3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c63a9eee8645207e0774315cd3135d29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c63a9eee8645207e0774315cd3135d29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6fe32dd9363fab09c82b7c46e07356a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4aab61e9523361c3d72e2da1849bdc52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff41168956ad907504cfd95d5c6c3c91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2031422b2206bea80dde1a8be626ed6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0863b19feb690bf3aedb7844d2fae799(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ff750e77bd5cfcddfa8741d65db6cff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.097085952758789]], [[6.046693325042725]], [[5.884105205535889]], [[4.725864410400391]], [[5.701006889343262]], [[5.194499969482422]], [[5.5775556564331055]], [[5.749197006225586]], [[5.571490287780762]], [[6.124773979187012]], [[6.355926036834717]], [[5.778453350067139]], [[5.824944496154785]], [[5.144975662231445]], [[6.082563877105713]], [[6.516267776489258]], [[6.008426666259766]], [[5.67205810546875]], [[5.723589897155762]], [[6.531248569488525]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af66697ab604a97b837f36bf0aa53510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 320, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([17, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ebdd17a22afbeabae7eb86d6b5c514b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 600, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 600, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ebdd17a22afbeabae7eb86d6b5c514b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 600, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 600, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_62b302a200ca87219f38f5a8b118e4c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2127dcfe353b315b8bac713817a5c27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 768, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e7c9f866b9e94d5b5b0243f56e9aecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a4c33f6c57c9734e07f0648ac07e432(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d63b68d16b1cb7aa74240061834715b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([6, 256, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9abb18844e20427c2b19fd9f99857ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75315a28ae38a2d9a225e82e1912c75a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9abb18844e20427c2b19fd9f99857ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d21622739e9fdc656e5bfc63bcd4ee5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 480, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_324056cf3b9a340265120bf8dc436b7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_16344879cd62c0b18879f0ce030fd952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34cb8bd6b44f38073b2bd90b5c557998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 56, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_324056cf3b9a340265120bf8dc436b7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_16344879cd62c0b18879f0ce030fd952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f612aa72a1c13c435435aa7da26c3298(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_324056cf3b9a340265120bf8dc436b7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_16344879cd62c0b18879f0ce030fd952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a4b76b27f07d324ed0a0736fd8ae822(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 16, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_324056cf3b9a340265120bf8dc436b7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_16344879cd62c0b18879f0ce030fd952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1d90698b61f69df254cb888d4f57472(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90dfb4a5d0a3c4c81595856e5d77a898(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a732ae88e398a6b024074f74ae710a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[34353.5546875]], [[38394.26953125]], [[34200.40234375]], [[31302.564453125]], [[43417.86328125]], [[30792.943359375]]], [[[34845.12109375]], [[38938.72265625]], [[34679.55859375]], [[31738.2265625]], [[44033.21484375]], [[31231.359375]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_401a63f8ff8017ddbdf0ccf4138c0975(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90dfb4a5d0a3c4c81595856e5d77a898(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e67cfa32ca5930554c7b41653a0acd50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[42160.09765625]], [[43356.0234375]], [[34807.3203125]], [[41353.58984375]], [[43749.4765625]], [[40938.83203125]]], [[[40549.54296875]], [[41706.1171875]], [[33480.37890625]], [[39771.7265625]], [[42081.734375]], [[39377.5703125]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_33f3f3bdb1a1bdb806426e8d12c50269(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90dfb4a5d0a3c4c81595856e5d77a898(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67aae0ac5a65366de4c2e706f3fb5801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[37815.68359375]], [[46491.82421875]], [[37637.91796875]], [[36063.03125]], [[46331.71484375]], [[46617.09765625]]], [[[36356.3515625]], [[44704.7265625]], [[36185.62890625]], [[34671.5625]], [[44551.7734375]], [[44829.05859375]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef123a9ddbe26f58a5897718249dd597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90dfb4a5d0a3c4c81595856e5d77a898(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1cfb064044b1534ad29f0c24e680ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[46384.93359375]], [[41454.01171875]], [[37989.71875]], [[40664.18359375]], [[45870.71484375]], [[38020.32421875]]], [[[44348.6796875]], [[39629.921875]], [[36315.76171875]], [[38875.1171875]], [[43854.77734375]], [[36348.61328125]]]], dtype='float32').reshape([2, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8c7563edf4be58dbed904e7ba92e35a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a17d192ffcdf81ace250d40613fd910(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1152, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7474d5f6e25494524a4c8e0332527f68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da37eea445df6759ec5737fbd9b8df56
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_394c2249f770584d88e204d4d2ec0bbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_394c2249f770584d88e204d4d2ec0bbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_394c2249f770584d88e204d4d2ec0bbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_394c2249f770584d88e204d4d2ec0bbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_394c2249f770584d88e204d4d2ec0bbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_394c2249f770584d88e204d4d2ec0bbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_394c2249f770584d88e204d4d2ec0bbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_394c2249f770584d88e204d4d2ec0bbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14574b281818180fbc03e6db77ed289d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_42c893cb78627f02a89d3967aa65a4f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffc37705d099e6eb65f4f10f83224a8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffc37705d099e6eb65f4f10f83224a8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffc37705d099e6eb65f4f10f83224a8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffc37705d099e6eb65f4f10f83224a8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffc37705d099e6eb65f4f10f83224a8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffc37705d099e6eb65f4f10f83224a8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffc37705d099e6eb65f4f10f83224a8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffc37705d099e6eb65f4f10f83224a8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_212fb3409ce2d346674ad169ab562892(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_509f0fec2f3e1ae8763265419a466d3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5aeaea6d467e9980b248248fb484609(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5aeaea6d467e9980b248248fb484609(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5aeaea6d467e9980b248248fb484609(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5aeaea6d467e9980b248248fb484609(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5aeaea6d467e9980b248248fb484609(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5aeaea6d467e9980b248248fb484609(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5aeaea6d467e9980b248248fb484609(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5aeaea6d467e9980b248248fb484609(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95d42e1d9c924edcbaee2331846d643f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8303c2d7bb11819baaac7264632ffd41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0eab51b7ff427f56fed4a20f675473c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0eab51b7ff427f56fed4a20f675473c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0eab51b7ff427f56fed4a20f675473c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0eab51b7ff427f56fed4a20f675473c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0eab51b7ff427f56fed4a20f675473c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0eab51b7ff427f56fed4a20f675473c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0eab51b7ff427f56fed4a20f675473c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0eab51b7ff427f56fed4a20f675473c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52c2195606f1e6fd9876d8fb247e9f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ff95a970455b05f5aad5f2180a2bbc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6bf42bbf132e41c70bb69f6b3231e82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6bf42bbf132e41c70bb69f6b3231e82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6bf42bbf132e41c70bb69f6b3231e82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6bf42bbf132e41c70bb69f6b3231e82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6bf42bbf132e41c70bb69f6b3231e82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6bf42bbf132e41c70bb69f6b3231e82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6bf42bbf132e41c70bb69f6b3231e82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6bf42bbf132e41c70bb69f6b3231e82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_485f873e67b989ef5ab51e947bf01117(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc8356623c8763e351a8985ea1138dc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a9b0ed433bceca2b272fcdab7cece3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_576da73f0f28d153d4f7fa1f448c22cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72044c3d524d866805bb343969ed6253(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 258, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce3318ea808029f56edac1ab79709fe9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f377986ce9aed23ad2a17f8d645b8a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_998923a7b5c09a40317897b57129526e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 258, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a1641833f64bf816e529b4a9bd63782(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 9261, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.18874695897102356]], [[0.203751802444458]], [[0.444195955991745]], [[0.1276213526725769]], [[0.103150874376297]], [[0.07216836512088776]], [[0.3980652391910553]], [[0.36862778663635254]], [[0.054698314517736435]], [[0.23712150752544403]], [[0.27820244431495667]], [[0.11455938220024109]], [[0.111771360039711]], [[0.05471891909837723]], [[0.4878503382205963]], [[0.1512620598077774]], [[0.36344748735427856]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_104aea0bc030fe82718cc1993895e10b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a6213285df22ec11f0bff694328a8015(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b5270960752a045a223886347bb562c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58d94e3d090f30e75dd90c1b1be35f1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb1bf76394699ce112821c53c47ac5ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.219970703125]], [[7.784126281738281]], [[7.827428817749023]], [[8.209232330322266]], [[7.692371845245361]], [[7.626750469207764]], [[8.033778190612793]], [[7.528597354888916]], [[7.796234130859375]], [[7.446854591369629]], [[7.121368885040283]], [[7.051188945770264]], [[7.789116859436035]], [[7.609338283538818]], [[6.6510443687438965]], [[7.625375747680664]], [[8.255825996398926]], [[8.235020637512207]], [[7.581562519073486]], [[7.541038990020752]], [[7.979496002197266]], [[7.4575347900390625]], [[7.476142406463623]], [[8.075897216796875]], [[7.466970443725586]], [[8.125590324401855]], [[7.947990894317627]], [[8.003846168518066]], [[7.388821601867676]], [[7.886233806610107]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7565a34eac33399053cc90c8b1f8ea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58d94e3d090f30e75dd90c1b1be35f1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ada1f3bbd2191c668e23f2d7b956eab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.814724445343018]], [[6.774284362792969]], [[7.524858474731445]], [[6.822107791900635]], [[7.457672119140625]], [[7.345535755157471]], [[7.476398468017578]], [[7.1589579582214355]], [[7.322680950164795]], [[7.283619403839111]], [[7.031378746032715]], [[6.672905445098877]], [[6.4484171867370605]], [[7.1944966316223145]], [[7.280976295471191]], [[7.0551629066467285]], [[7.122607707977295]], [[7.416973114013672]], [[7.247720241546631]], [[6.785580635070801]], [[6.085677146911621]], [[6.414621353149414]], [[6.927490234375]], [[6.715959072113037]], [[6.866250038146973]], [[6.130156517028809]], [[6.598968505859375]], [[6.8014678955078125]], [[7.103934288024902]], [[7.52005672454834]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0a597c74741a4150682b20cd8c6f4fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a365da5875496dbcb8f739416089598f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e562011eb901b481832d93e3a68b645(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58d94e3d090f30e75dd90c1b1be35f1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4d3cd0b169652ed828497c94267e45a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.571072578430176]], [[7.910855293273926]], [[7.639548301696777]], [[8.479670524597168]], [[9.00108814239502]], [[8.445219993591309]], [[7.997985363006592]], [[7.674211025238037]], [[8.481038093566895]], [[8.052382469177246]], [[8.052838325500488]], [[8.102134704589844]], [[9.469754219055176]], [[7.3469061851501465]], [[8.879927635192871]], [[7.816487789154053]], [[7.959301471710205]], [[8.373373985290527]], [[7.350678443908691]], [[8.218818664550781]], [[8.802421569824219]], [[7.801962852478027]], [[8.044658660888672]], [[8.281781196594238]], [[8.116142272949219]], [[8.111846923828125]], [[8.50810718536377]], [[9.174527168273926]], [[8.281989097595215]], [[7.9975972175598145]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9cccd4885fde2f19b3797f5645333872(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([50, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eaa52dc382708096de2af1adbe711709(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([200, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1a0b8af0dcbe498293df9c913d472a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3b2d7fe79395d6fd7390e41fc929135
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 4, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fccff50a46097a162d08ee87efa6791f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 136, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3afa4a94e5d1bf1ae4759ae0b4eab768(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 68, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb5598353038957d88641af0e82d315d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 34, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5dd76389d31bdc8b5693058de7b485fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 17, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57fc0a9f19331f71199c2f54480142f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 136, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6bbe5a0d81fdcae83afd9773c5ea5eef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 68, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd898d4c0eaacb983dc73450de60879a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 34, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_694f9c986005274030fe0a3d488ac9f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 17, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0103daf911d67970957134e07593115d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f8697189f9859721a8a7068ec7d8ab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 544, 640], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 3, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83f4289d5cd5d97657290345af13c52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d9da517f8efc16f3d2fb2af0b496a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3847b2a234db474cfde70666e9166af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1568, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3027c96c773fa4e834758d11579658f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3027c96c773fa4e834758d11579658f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3027c96c773fa4e834758d11579658f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dbb76248fbf5c932840d3dc1424d3fde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9abb18844e20427c2b19fd9f99857ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58d94e3d090f30e75dd90c1b1be35f1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ca022e8386956133bb219c177df7299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.1592535972595215]], [[8.019697189331055]], [[9.132721900939941]], [[8.245113372802734]], [[8.201904296875]], [[8.027078628540039]], [[7.732528209686279]], [[8.434311866760254]], [[8.001360893249512]], [[8.272842407226562]], [[8.75783634185791]], [[8.043542861938477]], [[7.47999906539917]], [[8.129373550415039]], [[9.145278930664062]], [[7.5090179443359375]], [[7.697664737701416]], [[8.60932731628418]], [[6.7332634925842285]], [[8.00938892364502]], [[8.664624214172363]], [[8.28862476348877]], [[7.382583141326904]], [[8.051594734191895]], [[7.450173854827881]], [[8.692798614501953]], [[8.793678283691406]], [[8.495599746704102]], [[7.484212875366211]], [[9.080621719360352]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f690e187d6d514c00cd5380ef27eba46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 288, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc5a4d89f99fc5d66e13857b732f72ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([12, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_33a20d1a5fbce4bae97cbd647ac94e90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.03708553314209]], [[3.342827081680298]], [[3.163536548614502]], [[3.6480274200439453]], [[3.1267178058624268]], [[3.629101276397705]], [[3.069321870803833]], [[2.996713161468506]], [[2.89133882522583]], [[3.521888494491577]], [[3.979022979736328]], [[3.150074005126953]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.uniform([48, 12, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d60c074af5424cd64fbe63174f59591f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f8697189f9859721a8a7068ec7d8ab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_77d703e90bc20d979024d57fa72cc8b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc5a4d89f99fc5d66e13857b732f72ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([12, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec0ca5015acaa5b9ea0443a0cb439819(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.799421787261963]], [[2.518261671066284]], [[2.8721373081207275]], [[2.9185006618499756]], [[2.6263015270233154]], [[2.7743587493896484]], [[3.625351667404175]], [[3.072920322418213]], [[2.7595460414886475]], [[2.802654981613159]], [[2.82415509223938]], [[3.3745367527008057]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.uniform([48, 12, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b54285e85a5ad2cfb2f2271470a7fcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98b7731f6613b2c4b1fe0029fed3815d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb1f61cbb788f8970388395fe3b9e4df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02850010e9988e9d71993bb2bdf38948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d91ed53660769be675e12511cbe0a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5bb26411a5a9466967040514bad77e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 76, 116], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7657effe283b639cb9527f3d29aeeabe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 76, 116], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dcb174e9b7b17c2aa6dcf42512fd4793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([25, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_051249151a50d24810add74b3c955369(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.523202419281006]], [[6.377389430999756]], [[6.686032295227051]], [[5.554091453552246]], [[6.187918186187744]], [[5.89094877243042]], [[5.716607570648193]], [[6.461143493652344]], [[6.702357769012451]], [[6.33857536315918]], [[6.324361324310303]], [[6.188377857208252]], [[6.650539398193359]], [[6.015854835510254]], [[6.307999610900879]], [[6.638586044311523]], [[5.999324798583984]], [[6.248127460479736]], [[5.995881080627441]], [[5.776070594787598]], [[6.150759220123291]], [[7.176509380340576]], [[6.003063201904297]], [[6.2718658447265625]], [[6.431493282318115]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.uniform([100, 25, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89f7459b87508b092fa3feb3089aada9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 320, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_00febfd0f86ef30c06a953c066f44ef5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_49349b6105717e92618bb871250a7115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([72, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c391da67c8299062e7379ab3e8be9bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([288, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e600e591d5c517e23a173812900c2b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd651aa747d74dfe7899adc192f6b0a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 3840, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 3840, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9abb18844e20427c2b19fd9f99857ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e8bea90c4824abb2f2ed50f587929cc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([12, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e8bea90c4824abb2f2ed50f587929cc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([12, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d1931eba524fdfe8a5d0bec6a7397f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31eadf7d89f032f040b5f2953e43ec43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 2100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4767546057701111]], [[0.3394171893596649]], [[0.1315995305776596]], [[0.35266992449760437]], [[0.3209752142429352]], [[0.3108814060688019]], [[0.05758047103881836]], [[0.42704105377197266]], [[0.23024608194828033]], [[0.0861610695719719]], [[0.1720646470785141]], [[0.018488945439457893]], [[0.2748638987541199]], [[0.44882333278656006]], [[0.35698819160461426]], [[0.3654640316963196]], [[0.359065979719162]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_dbe49d935de57f88926a9cc8a832f0fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dbe49d935de57f88926a9cc8a832f0fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b522deadd0b9f08bad4bba7b584ad3dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e4b6734851235fae328d214d1d06f1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([21, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e4b6734851235fae328d214d1d06f1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([21, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e4b6734851235fae328d214d1d06f1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([21, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf91f020746d6ab7b0d6da94c16d65e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([21, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0eaa37e5d395dda62f90402103de4242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([288, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1537cc8a04566e9e215e4839f6431a4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1537cc8a04566e9e215e4839f6431a4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09b9302ca5d134cc496435574e71c602(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c820e06d4096aa67457255ce55132a1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d91ed53660769be675e12511cbe0a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31146a2bd8bbe0b6f79553b97faafa0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5339846a14848a243467c6574e5bf7bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ccc41c5d0f52b34fba07d0e01b141a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b4a61366380b3f90d58aab4a617311c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d70594a75016a091bddf0e1ec973842(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_51b8f03b0600ede77d25dbfaf2c471ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d51198f417b715876583fb22df9b53f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_170e1f11ec20bbf4156617a2b70940e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b461e174da9fe18b73ef37cf5865d6e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_831b1bb8679d91d2d2c999f383652692(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a218af7299ea3a6b19b60e08f931a1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_924cc96d3d51e9ebf0b38366cd3eaadd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ce04d639b8a8b6facf7c5f513cdb239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_12dc59f4912a4d99664558ed028c1691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.849146366119385]], [[5.562366485595703]], [[5.357488632202148]], [[4.600664138793945]], [[5.43474817276001]], [[5.1333842277526855]], [[4.804915904998779]], [[4.552052021026611]], [[4.770118236541748]], [[5.240309238433838]], [[4.955009937286377]], [[4.486055850982666]], [[4.7744245529174805]], [[5.813625335693359]], [[5.323536396026611]], [[4.844731330871582]], [[5.070008754730225]], [[4.528096675872803]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7bd1196425248ed8741ebf0b85b15e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87a96aa1829ec132f7ca4fbd3c9fe68f
    def get_inputs(self):
        return [
            paddle.uniform([4, 3, 384, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 3, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01c8bddbf308a5860d2d145389392de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d1931eba524fdfe8a5d0bec6a7397f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a493dc5bb4b1a466a9d77a4d75151c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 258, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d91ed53660769be675e12511cbe0a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1dcac7696a34f448556cb41d0172721(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([768, 384, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c9ec1cec7d3df82d2e61074cc0e3f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b6792f8e37c162942514c323017c8cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c80975984dc592ac26b2b58adef0117c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6503349600bcfd179b3e2ea0d5741d5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6503349600bcfd179b3e2ea0d5741d5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1cc6de9811de27649d5dfbd45bf6d03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6beb8d7235ccfb5077882c8f8ce66a22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.056128378957509995]], [[0.28628548979759216]], [[0.036558814346790314]], [[0.2744646370410919]], [[0.24790284037590027]], [[0.3711458146572113]], [[0.26032090187072754]], [[0.4726661741733551]], [[0.4423083961009979]], [[0.0875990092754364]], [[0.2537902593612671]], [[0.1440911740064621]], [[0.4323233664035797]], [[0.02738807536661625]], [[0.4764312505722046]], [[0.14398229122161865]], [[0.10762173682451248]], [[0.013186508789658546]], [[0.19020965695381165]], [[0.30745774507522583]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([5, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3abbecb117aee499a648ec1e11717dbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3530144691467285]], [[1.4647397994995117]], [[1.5248286724090576]], [[0.8225886821746826]], [[1.303797960281372]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.uniform([20, 5, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a85896acf06ed79b107f6fbed029208a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4608daa6f95bfd06ebfeb86d4e5575a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.0290560722351074]], [[2.792099714279175]], [[3.0667827129364014]], [[2.414238929748535]], [[2.9861397743225098]], [[3.1318516731262207]], [[3.273455858230591]], [[2.80271315574646]], [[2.6938159465789795]], [[2.9530866146087646]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.uniform([40, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0863b19feb690bf3aedb7844d2fae799(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38f495c7b3dab94e4655b3ec1872fa42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.387393951416016]], [[5.253484725952148]], [[4.903465747833252]], [[5.175287246704102]], [[4.975003719329834]], [[4.971848487854004]], [[5.1791768074035645]], [[5.112245082855225]], [[5.15993595123291]], [[4.282576084136963]], [[5.401867389678955]], [[4.462133884429932]], [[5.145967483520508]], [[5.319954872131348]], [[5.04836893081665]], [[5.1379265785217285]], [[4.489475250244141]], [[5.0134477615356445]], [[4.9873480796813965]], [[5.273707866668701]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4aab61e9523361c3d72e2da1849bdc52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff41168956ad907504cfd95d5c6c3c91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d91ed53660769be675e12511cbe0a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d91ed53660769be675e12511cbe0a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9cccd4885fde2f19b3797f5645333872(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([50, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eaa52dc382708096de2af1adbe711709(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([200, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ff6d94a5f4992d259f31063ce7101b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cbacdf74fbf18665bb80e8d36bd3964(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_544dd291627d3a5f491850d977d08638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_967ad17312112cf9ca062c6a2ea45909(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ff6d94a5f4992d259f31063ce7101b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e7f49745d998a6056700a94908815411(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ea2277c73ce3e08991c1a36263938193(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 192, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55ca60a55a7e3273eec550964208bb6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bcd9a220685365281e86d4874d730b51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 4, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03bdd818623d9c3c04c312d660fd63c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_930041cf69db0eb68add516aa3d99c62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_933757c1fd0a8b4d60131547efdeb459(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_408d50dd31f8f64474a164935618d860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c96f905d1208e391dc36611aabce14db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_912018ef568afbe72e6d0cf02f65e70e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d0d65e7dd6fe3f850f1a76da08ac994(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c96f94953ae196d6bb9b63188d302a67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a9b0ed433bceca2b272fcdab7cece3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_576da73f0f28d153d4f7fa1f448c22cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7b865535c86af54f8e09c7e88c40a8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_49cb161d0b7bfaa8cb0fabb27b497ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f72919153772386b1981e3bb12c513d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_417ef176a25f2f6f0ee3aacce164b341(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 11109, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2855564057826996]], [[0.2848183810710907]], [[0.27559658885002136]], [[0.4871450960636139]], [[0.34324249625205994]], [[0.4266868531703949]], [[0.25881263613700867]], [[0.11944302171468735]], [[0.020073212683200836]], [[0.1890229731798172]], [[0.4114799201488495]], [[0.26791492104530334]], [[0.04340853542089462]], [[0.09969694167375565]], [[0.058066412806510925]], [[0.09210151433944702]], [[0.19802170991897583]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_9abb18844e20427c2b19fd9f99857ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5586349df1449ff47eeb1060d89834fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e545c24eec88e86ab3a221d516183b1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.206503868103027]], [[7.111752986907959]], [[6.231878280639648]], [[5.649900436401367]], [[7.215942859649658]], [[6.2238922119140625]], [[5.985706329345703]], [[6.53260612487793]], [[7.171493053436279]], [[6.2621378898620605]], [[6.22059965133667]], [[6.320767879486084]], [[6.109984874725342]], [[6.1065568923950195]], [[6.8695759773254395]], [[6.2865705490112305]], [[5.723404407501221]], [[6.720170974731445]], [[6.197271823883057]], [[6.079874038696289]], [[6.363616943359375]], [[6.1645965576171875]], [[5.554739475250244]], [[6.695564270019531]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c459d2fff77f56c923c3999ef167030b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([180, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c459d2fff77f56c923c3999ef167030b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([180, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c61b9d95ea4838915ef0831c20883822(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([392, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a85896acf06ed79b107f6fbed029208a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04101fab825ae5395836932db92e7577(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.351255178451538]], [[3.0869038105010986]], [[2.3032050132751465]], [[2.3017141819000244]], [[2.4560351371765137]], [[3.090928077697754]], [[2.6265499591827393]], [[2.695026159286499]], [[2.9914133548736572]], [[2.802804946899414]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.uniform([40, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cce92b90ff4499143e35b36796012af5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6838fb5260105e08f6049e3815e80ee3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7e308ee428732cfc424d68bf94de633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98f694a7afffc8b7b82ee32dedd06592(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2bf064b21c5e32abdc41f81b2147813(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8991941ab6425a50abc5bf3f23bc1173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc531b285c2283c40ca50e2d449b1e8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_725c5dc03f3655fff8db12430eae8451(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ae7696652ffcfd9752fb22d09b6713c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 144, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0b7e50391c8640da1869ecc1a291b9d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ee7189c46e8c23abff2b840814aa260(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f8c310c5c74159b467c6472c365dc8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2159675c24e5b3b1559def87e146f001(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b6ae422c72ee1081ae9dc3a720d086b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c9fd701dc9b972e36a0131e313d61acd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([2, 256, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6baf9f5b4c6d5eab5a1b5f915d40ea12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01d5108473eaeb63359483767618eabb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9abb18844e20427c2b19fd9f99857ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6b8d9d0a68097906fad2324bde1a1028(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([3, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d520e04d26f949730816e6b231438a85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a9b0ed433bceca2b272fcdab7cece3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_576da73f0f28d153d4f7fa1f448c22cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_24b228454f035ff625bf40a5f3bd7603(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([150, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ce04d639b8a8b6facf7c5f513cdb239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4108a9162dca6c45dd84729b410d74ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.880243301391602]], [[4.284125804901123]], [[5.1647257804870605]], [[5.034191608428955]], [[4.1328043937683105]], [[3.989349365234375]], [[4.830134868621826]], [[3.9868626594543457]], [[3.904547929763794]], [[4.827010154724121]], [[3.8705732822418213]], [[4.75415563583374]], [[3.9656474590301514]], [[4.086240768432617]], [[4.686644077301025]], [[4.369600296020508]], [[4.487377643585205]], [[4.246503829956055]]]], dtype='float32').reshape([1, 18, 1, 1]),
            paddle.uniform([72, 18, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0eaa37e5d395dda62f90402103de4242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([288, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f7375bbcbd486cde7cbfe4ab32faa7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 92, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2fac65463bbbad87fe5b9568afe087b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1568, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fde96a941964bc377f551203449eb7f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([21, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efbdca9d353a0dc70a999c0eab6c6d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a70f678450ee47c9942ce2efb794697b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08acc4a9b0648a236696ce8a9858b420(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 32, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 320, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57a83fd09d3d6d5bca43ca7ba2ae8511(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89e04cc010900e2df8f969b5a2ec9ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a091cb92fc718ce7e89f1ff9c4974ef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6fa1db336b991e4da73814c8da9abe38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a86c343f0598b559276b35fc8e1f8974(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_710d2a51cda281c278072e1d468f4561(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_394c2249f770584d88e204d4d2ec0bbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffc37705d099e6eb65f4f10f83224a8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5aeaea6d467e9980b248248fb484609(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ff6d94a5f4992d259f31063ce7101b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1413af3754b05e6abc8cec4fdb831044(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 258, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df37fd8d49be17ad89a87acf9f2b7a02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 32, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 160, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66f40cf2517b2ac1a155f09bc331afb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66f40cf2517b2ac1a155f09bc331afb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efbdca9d353a0dc70a999c0eab6c6d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a70f678450ee47c9942ce2efb794697b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cbb3798140e6c41939719f80e52313e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_00febfd0f86ef30c06a953c066f44ef5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58d94e3d090f30e75dd90c1b1be35f1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4176896da54fb9075bcb37b336a29108(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.62192440032959]], [[7.400206089019775]], [[7.748739719390869]], [[7.875727653503418]], [[7.208639144897461]], [[7.998118877410889]], [[7.931631088256836]], [[7.8193678855896]], [[7.174508094787598]], [[8.173986434936523]], [[8.253493309020996]], [[7.992372989654541]], [[7.904558181762695]], [[7.437554836273193]], [[7.751658916473389]], [[8.34387493133545]], [[8.273571014404297]], [[7.511903285980225]], [[7.043776512145996]], [[8.035367965698242]], [[7.823747634887695]], [[7.011332035064697]], [[7.618360996246338]], [[7.738470077514648]], [[8.361632347106934]], [[7.775673866271973]], [[7.732045650482178]], [[8.275997161865234]], [[8.61551284790039]], [[7.694608211517334]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_916a6654bc180836eb3af9a42544a6d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.1613265872001648]], [[0.05861525982618332]], [[0.19764043390750885]], [[0.05961800366640091]], [[0.22183170914649963]], [[0.08580177277326584]], [[0.2509969174861908]], [[0.43331530690193176]], [[0.11445356160402298]], [[0.296318918466568]], [[0.002657710574567318]], [[0.19037719070911407]], [[0.11254051327705383]], [[0.19441959261894226]], [[0.47737017273902893]], [[0.03603414073586464]], [[0.4848943054676056]], [[0.18987561762332916]], [[0.4618443548679352]], [[0.4588715732097626]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([5, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0c83db5c880bd39d5474ad7197f9a37f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.7743725776672363]], [[1.4237773418426514]], [[1.4726775884628296]], [[1.4478960037231445]], [[1.8577220439910889]]]], dtype='float32').reshape([1, 5, 1, 1]),
            paddle.uniform([20, 5, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a85896acf06ed79b107f6fbed029208a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d95c6dd1cf799710f58605878134ae5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.6523306369781494]], [[2.4763336181640625]], [[3.2539005279541016]], [[2.871127128601074]], [[3.1752514839172363]], [[2.687026262283325]], [[2.6177079677581787]], [[2.592658042907715]], [[2.8259291648864746]], [[2.734337329864502]]]], dtype='float32').reshape([1, 10, 1, 1]),
            paddle.uniform([40, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0863b19feb690bf3aedb7844d2fae799(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_108e0ead748d446342e9677c68e1b7bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.181726932525635]], [[5.7877678871154785]], [[6.224140644073486]], [[5.852336883544922]], [[4.836845874786377]], [[5.370150566101074]], [[5.751057147979736]], [[5.799452781677246]], [[6.094420433044434]], [[5.716272354125977]], [[5.1642937660217285]], [[5.500518798828125]], [[5.989058494567871]], [[4.902487277984619]], [[5.668533802032471]], [[5.383454322814941]], [[5.875471591949463]], [[5.620500564575195]], [[5.334749221801758]], [[5.4819560050964355]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_77b9607fc40ba1e59e94fa712b27c0a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55ca60a55a7e3273eec550964208bb6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bcd9a220685365281e86d4874d730b51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 4, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4aab61e9523361c3d72e2da1849bdc52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff41168956ad907504cfd95d5c6c3c91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_49cb161d0b7bfaa8cb0fabb27b497ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f72919153772386b1981e3bb12c513d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_924cc96d3d51e9ebf0b38366cd3eaadd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e7c9f866b9e94d5b5b0243f56e9aecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a4c33f6c57c9734e07f0648ac07e432(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7c6d452534dbacb41e69ffc852fd631(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b35e4b63cacc40a2a3cfbfacb8d7b701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.760829210281372]], [[4.256175994873047]], [[4.550135612487793]], [[4.613018035888672]], [[5.112522125244141]], [[3.745678186416626]], [[4.642297267913818]], [[4.18134069442749]], [[3.7590386867523193]], [[4.86866569519043]], [[4.42974328994751]], [[3.9300332069396973]], [[4.057292461395264]], [[4.5839738845825195]], [[4.189048767089844]], [[3.859541416168213]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ef3fa52ff5e450d562c3a76116b2a67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7433316928817df0f664eb874cdb1eac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54e41afc270c44b84b7f016b826e72fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_725db1397e23a27137f982417abdb9fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c61b9d95ea4838915ef0831c20883822(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([392, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5978f62e886ea7e84bf5fcc58e72c81f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fc459f994d642faf0c96e0cd4078b3ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a0810dca3ed28edde03e00ed56ef3976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2dbcc4718dd499c44d8f84df59ae4c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2dbcc4718dd499c44d8f84df59ae4c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2dbcc4718dd499c44d8f84df59ae4c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2dbcc4718dd499c44d8f84df59ae4c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2dbcc4718dd499c44d8f84df59ae4c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2dbcc4718dd499c44d8f84df59ae4c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2dbcc4718dd499c44d8f84df59ae4c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2dbcc4718dd499c44d8f84df59ae4c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab97a957fe48aad33df5e4df0156f302(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d04744c339d126ffaa392e9d47eacc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfc1eb7b5132e4da8d731b025b56af24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfc1eb7b5132e4da8d731b025b56af24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfc1eb7b5132e4da8d731b025b56af24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfc1eb7b5132e4da8d731b025b56af24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfc1eb7b5132e4da8d731b025b56af24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfc1eb7b5132e4da8d731b025b56af24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfc1eb7b5132e4da8d731b025b56af24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfc1eb7b5132e4da8d731b025b56af24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_37021b6880580ce3cd9e84fee3f76bcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e32a5a4b6356306aba53a94a50be2ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_906fb922b198b206659f95c05e6259c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_906fb922b198b206659f95c05e6259c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_906fb922b198b206659f95c05e6259c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_906fb922b198b206659f95c05e6259c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_906fb922b198b206659f95c05e6259c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_906fb922b198b206659f95c05e6259c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_906fb922b198b206659f95c05e6259c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_906fb922b198b206659f95c05e6259c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9176ac8eae6ab8ea92e9566536b3ca85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9487ce2f6dbad1a8aac45dcf40392975(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0eab51b7ff427f56fed4a20f675473c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0eab51b7ff427f56fed4a20f675473c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0eab51b7ff427f56fed4a20f675473c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0eab51b7ff427f56fed4a20f675473c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0eab51b7ff427f56fed4a20f675473c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0eab51b7ff427f56fed4a20f675473c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0eab51b7ff427f56fed4a20f675473c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0eab51b7ff427f56fed4a20f675473c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52c2195606f1e6fd9876d8fb247e9f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ff95a970455b05f5aad5f2180a2bbc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6bf42bbf132e41c70bb69f6b3231e82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6bf42bbf132e41c70bb69f6b3231e82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6bf42bbf132e41c70bb69f6b3231e82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6bf42bbf132e41c70bb69f6b3231e82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6bf42bbf132e41c70bb69f6b3231e82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6bf42bbf132e41c70bb69f6b3231e82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6bf42bbf132e41c70bb69f6b3231e82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e6bf42bbf132e41c70bb69f6b3231e82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_485f873e67b989ef5ab51e947bf01117(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc8356623c8763e351a8985ea1138dc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4f893af74a8dad72d95c5a36548df34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 12, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4f893af74a8dad72d95c5a36548df34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 12, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04de6984e95cce902c0fa64dee1a53f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7e476089499b9ea97a7f6822aa343a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([336, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b175310087d2e9aca4c6b884e2c22af3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17f0b70978afb18f58ebccbfed08d1bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06d965ad93e9b3d2a9855fbe4a946706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_93efce908767c727bc56949a05f48f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 180, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_93efce908767c727bc56949a05f48f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([60, 180, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_88d2d9f7bda56f0fb075a6b5bab71054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([14, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d3d466e7730599b55667e87a2fb69cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.1301262378692627]], [[3.0191736221313477]], [[2.864104747772217]], [[3.18454909324646]], [[3.425471782684326]], [[3.2734270095825195]], [[3.5372228622436523]], [[3.3475706577301025]], [[3.380138635635376]], [[3.4204189777374268]], [[2.9306490421295166]], [[3.359321117401123]], [[2.9161040782928467]], [[2.9420838356018066]]]], dtype='float32').reshape([1, 14, 1, 1]),
            paddle.uniform([56, 14, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d677e45c453e7447987249d0767ec07e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([576, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11a2d5308766053aa79fc8af4dab96f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([91, 480, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bb7feac02113d053ee7705bf1b8357a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 144, 216], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_00dc8977f7919791203287c2477804c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 72, 108], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_661540b49ae68ca740cb2387dca715f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 36, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec7c7a28cbb71c8d6ea624045f2af59a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 18, 27], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17f4acc7f0e247edd4fafa052d30cf10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 144, 216], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05487d1b5b05660aeb1233fb431b02ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 72, 108], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eed5f03652933ebc1f81a5d35f75df5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 36, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c747f6c1c7afde96dfb10dc817562e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8507f8f1d78a3a2781c4a20682f133ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f8697189f9859721a8a7068ec7d8ab6
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39f25ada9122c917b0eb2440113c2ce3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ddccfd2085a6d1e0ae82acf87702de2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dab59acbd982d7cce588c03eaacfc831(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b8770b6e5da34956a213918afae56241(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af1e73158f235f5cfc598d034bca74b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_094e0766d62467c5b67609aa986de094(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_665c5e007740e67b88159863271ecc48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc8d13372a624e4c6b8f706fe76bc42c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_93863141924cb9f7d611884029801396(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_321c4b797e0ccc7af3fe0b339746ba8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 84, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3df46ec58aa75ed3668621417a722ba4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 42, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f773253e3893a96fc4ab203c1699d25d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 21, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eab7915c11b790cf605540da4ce463b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_46eb2fada483a31d1dab8d9622608bb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edff0a9c5142954163b1f61c589fa3ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd8d035a1f47367df7dffdbdb3b5971e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0863b19feb690bf3aedb7844d2fae799(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a910bcb76d8a134c080bfd4b152bab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.636407852172852]], [[4.719157695770264]], [[4.447378635406494]], [[4.884665012359619]], [[4.541313171386719]], [[4.4184441566467285]], [[4.3862786293029785]], [[4.829842567443848]], [[4.4603729248046875]], [[4.298841953277588]], [[5.239867687225342]], [[4.0581583976745605]], [[4.3401641845703125]], [[4.505803108215332]], [[4.745649337768555]], [[5.160227298736572]], [[4.51344633102417]], [[4.922163009643555]], [[4.5199103355407715]], [[4.295330047607422]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39fc09290faafdef6899576f2ed4cb18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39fc09290faafdef6899576f2ed4cb18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fbef0cb63b01e7f76322840113676b36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9137e7e8211ceca4b0fc1313e30e87cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 100, 136], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2de072d94d93c3231d343372eacacefa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 50, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba6fd9b553bc925063c4f5ab6de10791(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 25, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28b7bb743355b2a968c195d28b9cb769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae8eba19e226e56033da98cd2e47256f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 100, 136], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d4bf11d00752e126ccfa31b42a551f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 50, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb2e717d6380b21083998cab6c0214b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7c8ea89b4907eeb77a293ab1d58bda6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 192, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d34b3f956d0869a923f797c18decae70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d5fe686e3307ffe52ec6d77d3c20ee8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([384, 384, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba59db6316f2ef3c05fb5bedba026cd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_15e70c02203c45e2fe1e53fc55e4e0f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66cfcbe54eed6e2c82754833e3f44239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67b046711913a97cf2ecbd627d110aaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c84837658a4fc9597a950dbce5a73a38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c1739c9d08c00cefa9c5bdffc5ab223(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69eea2e5451e9c49220ebabb69ab22de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f9beaae57af3c8f5934c4ce7ac8cfa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_403553bba91352248978747218cd607a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 96, 2, 2], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_151f10c79c99129c9f7929c042778bf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a960c3fc46d9891d30083f9ba23ef19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cbacdf74fbf18665bb80e8d36bd3964(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_544dd291627d3a5f491850d977d08638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe0b71def3e95a2d97113dc40fbbc0f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d677e45c453e7447987249d0767ec07e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([576, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3a8b88446da276a5fc0b56f81e6deccf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 42, 42], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 480, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_164e64ea814953aaa25f3bc07536703d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58d94e3d090f30e75dd90c1b1be35f1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([30, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31ed3db5b09e6c7a6fbec6bff79974c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.801311492919922]], [[8.477471351623535]], [[7.692463397979736]], [[8.016641616821289]], [[7.590599060058594]], [[7.4950151443481445]], [[8.187170028686523]], [[8.017263412475586]], [[7.549017429351807]], [[7.662537574768066]], [[8.0874662399292]], [[8.07473087310791]], [[7.834475517272949]], [[8.067551612854004]], [[8.353202819824219]], [[8.677970886230469]], [[7.78714656829834]], [[8.194197654724121]], [[8.130566596984863]], [[7.409244537353516]], [[7.827935218811035]], [[8.826498031616211]], [[7.405838489532471]], [[8.205777168273926]], [[8.090394020080566]], [[8.595943450927734]], [[8.288196563720703]], [[7.462908744812012]], [[8.489489555358887]], [[7.518553733825684]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7868a303fac26c9f04f1a3b77fc27739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9e2eab064255804d0a8334b99e2ead2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([960, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d91ed53660769be675e12511cbe0a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4aab61e9523361c3d72e2da1849bdc52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff41168956ad907504cfd95d5c6c3c91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14b46dac3f20566056dae8179f18faaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03df4b0662fb64e0011d06b04d33f366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 768, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fc459f994d642faf0c96e0cd4078b3ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a0810dca3ed28edde03e00ed56ef3976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d7dc36206107c141aeffdaceaf25f4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f8697189f9859721a8a7068ec7d8ab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a5c66166bc7f810280f6f61eff72f94e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f8697189f9859721a8a7068ec7d8ab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 32, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e6ee55823199c82b05551890046e13d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([72, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_62be4ea7fae25aabba8cca819275b762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([960, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ae9cfa0e6e47753d779438b848df12d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            paddle.uniform([19, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d4d16cfd7a14e2f5ea263a710ca88346(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fea35457ca4ef53c4021b6d0e6c6239
    def get_inputs(self):
        return [
            paddle.uniform([22, 3, 224, 224], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 3, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf0ae6b82ad965159ffca6837b10d284(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ae071b18c65645d1fb3180c4cad8ffa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4bf27c01cc56fb23e90f8d8aa41e3702(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ece19002373f81afc4a051985765da89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ae071b18c65645d1fb3180c4cad8ffa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4bf27c01cc56fb23e90f8d8aa41e3702(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 16, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2da2197aaf9c8be2e6b27c87c207bde1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dac335706732c7464210335f428ddcb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b0deab019cb0dd19e6f6c6725721213(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 54, 54], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a81b164ee4a166879040de2d91471476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39b1012e083cfbf072fb70c1fd1250e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d3c8f751d9c658c704f849b1c6c53ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ec52518dae8523782ab12582a84d76f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cf77fe8af2e0a6dcd45928d628c27b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5dc2c837129610028a3c207a042f460(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07dd3eb2b1a321040be5f39b1eea3a54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7cf77fe8af2e0a6dcd45928d628c27b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5dc2c837129610028a3c207a042f460(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 48, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba0195503677f239cea80e4f0dd13e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13beb1068272307a9053e68c6613d2b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae9bce19e3e8b1b39424e8ff4acd42a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a457a7ea276c6d5f2924974edb321f9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_940a9e8a78606f47f268f0fb67f864bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4837110d6bea5091210a020cefd508d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_88f29b988f9ae12d510555aa47cfd4a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d91ed53660769be675e12511cbe0a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b9adececf44097218154795806d3600(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([4, 320, 16, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 320, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cd875e03cd7daf12a4084a1c207444c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4b21263f7b45facfb81f62e5ed19d8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da37eea445df6759ec5737fbd9b8df56
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_214ea0dbd87bda619f00c9daf6af0478(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_058afa0f0cb5d15344860e6f04ceedc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1cc04b9d84825d5375df29e05ba58bf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([4, 320, 8, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 320, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99b3e8160a7629b5c91499ad715b2a1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99b3e8160a7629b5c91499ad715b2a1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_820854f7a3ff276fe128583a5a3b2f48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c9e5ec4fbffab4b6cbd7e7705a81885(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 128, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9cccd4885fde2f19b3797f5645333872(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([50, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eaa52dc382708096de2af1adbe711709(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([200, 50, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_878d399d459efe4827540fd66596c2ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67187fc17cff812296a50f20abeb71e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0b5a7eb23accdcba1614849dead3d438(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8138ec3eb195bbcf57a94328a1a852d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34d437b42d9edf01c022acd75125f49c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3ccb04f1b52292f11b4f673eb16b7e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d6d3101fe417325228ca493bc334ce9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36d10eb9b7471805a910e4837d7d4922(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2c505983738be0b54b09b612f1419cd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 3024, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4462260901927948]], [[0.46795254945755005]], [[0.018542340025305748]], [[0.2093777060508728]], [[0.44989827275276184]], [[0.06751883029937744]], [[0.22981150448322296]], [[0.09533846378326416]], [[0.1647261083126068]], [[0.016285503283143044]], [[0.25232648849487305]], [[0.10952393710613251]], [[0.2692081928253174]], [[0.042666949331760406]], [[0.4003227651119232]], [[0.3115326166152954]], [[0.2864241898059845]]]], dtype='float32').reshape([1, 17, 1, 1]),
        ]


class TestPrimitiveOp_9178288e0627ec249ddd290e584ecea2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_531ea53ae7b65c13c8ba1792a7f7c191(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([400, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ff6d94a5f4992d259f31063ce7101b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f7b721473cafd0c119d539eab4abac7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ed311747efc0448d40a9c3c36400544(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5586349df1449ff47eeb1060d89834fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_33eb21a6f4167854b21e19bff731a1e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.737256050109863]], [[6.288643836975098]], [[7.323916435241699]], [[6.758274555206299]], [[6.437505722045898]], [[6.254459381103516]], [[7.311783790588379]], [[6.879544734954834]], [[6.972203731536865]], [[7.086414813995361]], [[7.219585418701172]], [[6.86547327041626]], [[6.434556007385254]], [[6.952660083770752]], [[7.1039958000183105]], [[6.371720314025879]], [[6.167044639587402]], [[7.03317928314209]], [[6.506866455078125]], [[6.954500675201416]], [[6.295381546020508]], [[7.109879493713379]], [[6.929393768310547]], [[7.029480457305908]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3847b2a234db474cfde70666e9166af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1568, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dcb174e9b7b17c2aa6dcf42512fd4793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([25, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0796ff8e160aa8e53a2842d0f7544cd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.119134426116943]], [[5.923222541809082]], [[7.408734321594238]], [[6.47472620010376]], [[6.32647180557251]], [[6.711030960083008]], [[7.998645305633545]], [[6.787867069244385]], [[7.9734272956848145]], [[6.716749668121338]], [[7.480202674865723]], [[7.065846920013428]], [[7.417253017425537]], [[6.721028804779053]], [[6.85184907913208]], [[6.518846035003662]], [[7.006399631500244]], [[7.119091510772705]], [[7.460700988769531]], [[6.329360008239746]], [[6.411314964294434]], [[7.972550392150879]], [[7.204898357391357]], [[6.429723262786865]], [[7.317016124725342]]]], dtype='float32').reshape([1, 25, 1, 1]),
            paddle.uniform([100, 25, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd3b2644896316d77a7b7b7fb7646fda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2498c8e446e41095cbb0510599e04e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 4, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc5a4d89f99fc5d66e13857b732f72ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([12, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef189ea4e0887f1f193a85d115816878(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.5534021854400635]], [[3.0459494590759277]], [[2.9850471019744873]], [[2.926222324371338]], [[3.026776075363159]], [[3.4852294921875]], [[3.641829490661621]], [[2.8090176582336426]], [[3.4768717288970947]], [[3.0273609161376953]], [[3.431544303894043]], [[3.369569778442383]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.uniform([48, 12, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0eaa37e5d395dda62f90402103de4242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([288, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2d94039c85ec465c362a5b76121fb3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b29297e18046f80a50897da31219c21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 8, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a9b0ed433bceca2b272fcdab7cece3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_576da73f0f28d153d4f7fa1f448c22cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4aab61e9523361c3d72e2da1849bdc52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff41168956ad907504cfd95d5c6c3c91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cbacdf74fbf18665bb80e8d36bd3964(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_544dd291627d3a5f491850d977d08638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([320, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0f440f6d3e12d52546f68281026e5333(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 76, 116], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c085c9d75bc6cb3c35f6ddfaafe48c29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 38, 58], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4fbc63391a49093eea3c1152e2204f5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 19, 29], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3a721955d89ed8e56c04f578f510b4b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 76, 116], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b42b2759cdd93f0fefb208976e1c8a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 38, 58], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b98b179fc8cc50e071572bdbbe2583a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 29], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b327d76c28c1c3c59c82dd1e6a3fae3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 29], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c96f94953ae196d6bb9b63188d302a67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e4b6734851235fae328d214d1d06f1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([21, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_643829a16fd6aefe9b602556716e316d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_011245b0b3d0394928343d4a12385d89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ef3fa52ff5e450d562c3a76116b2a67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7433316928817df0f664eb874cdb1eac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e66ef5ae5ff7b52de9dc71e38b36126(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 38, 58], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcad5ba3eff2eeaaf96affdf30b58bc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 38, 58], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_848561b68980c88503328128fc7393e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([112, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c9ec1cec7d3df82d2e61074cc0e3f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b6792f8e37c162942514c323017c8cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_74cc43dfaee8bed53fd3aca8ed269ab0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a017c6369133158ce1f64c91d9a379a
    def get_inputs(self):
        return [
            paddle.uniform([4, 160, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 160, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e57f92b202cf2479872f8007cb69e1ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c49767bebabcb2795df4800e2db7fac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17e6efe96cc6babde0b773ebe373c551(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4aab61e9523361c3d72e2da1849bdc52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff41168956ad907504cfd95d5c6c3c91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a9b0ed433bceca2b272fcdab7cece3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([120, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_576da73f0f28d153d4f7fa1f448c22cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([480, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_036c5a8ae5702386a17a17eb8782a1f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db6aaadabef5e4be6a6cbeb629e6cba6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c9ec1cec7d3df82d2e61074cc0e3f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b6792f8e37c162942514c323017c8cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 10, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ff6d94a5f4992d259f31063ce7101b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_164dae7c34f9ffcc06feb2d228bb661b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25d6195da08ab4acdddc6538063a09fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b175310087d2e9aca4c6b884e2c22af3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64315ed441451025ebc5275e25b21637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5586349df1449ff47eeb1060d89834fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db9d941317fbb644611bbdbde5751246(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[762.2291259765625]], [[786.2742919921875]], [[743.0218505859375]], [[697.415283203125]], [[699.0284423828125]], [[712.4510498046875]], [[749.775634765625]], [[764.63330078125]], [[686.6480102539062]], [[747.8648681640625]], [[747.8525390625]], [[758.94287109375]], [[732.6702270507812]], [[676.7434692382812]], [[756.5175170898438]], [[697.097900390625]], [[741.1431884765625]], [[777.205078125]], [[683.3544311523438]], [[776.3590087890625]], [[679.1268310546875]], [[784.6321411132812]], [[701.3018798828125]], [[703.9038696289062]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc922cbcbb8a0f05e495365bbfa2a236(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5586349df1449ff47eeb1060d89834fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89a7a9073c425cdb8f15829d11f665a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[89.13827514648438]], [[99.29498291015625]], [[104.95667266845703]], [[104.80584716796875]], [[103.9608383178711]], [[85.3790054321289]], [[97.73500061035156]], [[90.42082977294922]], [[97.99455261230469]], [[106.14417266845703]], [[100.25833892822266]], [[103.96082305908203]], [[94.44489288330078]], [[91.15996551513672]], [[104.29300689697266]], [[100.65071868896484]], [[103.41301727294922]], [[109.82306671142578]], [[92.8393325805664]], [[106.42680358886719]], [[100.84012603759766]], [[99.505615234375]], [[94.85469818115234]], [[98.58260345458984]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c26d8ecb328464ff79312da956efd66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5586349df1449ff47eeb1060d89834fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6494e6f039e824dbdfce5db002aeb42b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[47.50959014892578]], [[41.27824783325195]], [[42.36391830444336]], [[40.817386627197266]], [[44.72731399536133]], [[43.65764236450195]], [[46.42644500732422]], [[48.89083480834961]], [[45.73055648803711]], [[45.65370178222656]], [[44.44060134887695]], [[44.08531951904297]], [[44.43910598754883]], [[45.04508590698242]], [[38.68388748168945]], [[42.89270782470703]], [[43.38323211669922]], [[40.792789459228516]], [[42.09685134887695]], [[46.13134002685547]], [[45.7238883972168]], [[47.314598083496094]], [[47.06987380981445]], [[42.84346389770508]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed672a6fa958a7d75b2c393314ec40de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5586349df1449ff47eeb1060d89834fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf9604644235d24282781e3b29de80dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[21.57964324951172]], [[20.849185943603516]], [[21.389863967895508]], [[22.92589569091797]], [[21.12099838256836]], [[21.001768112182617]], [[21.905405044555664]], [[20.621068954467773]], [[20.97715950012207]], [[20.474952697753906]], [[21.735191345214844]], [[19.871593475341797]], [[20.53399085998535]], [[22.35291862487793]], [[20.863351821899414]], [[21.015045166015625]], [[21.126745223999023]], [[21.539566040039062]], [[19.160127639770508]], [[21.543785095214844]], [[19.366960525512695]], [[22.470134735107422]], [[19.518627166748047]], [[22.569242477416992]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f197a18a33ce0ca9b794aa3840a97a2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e300167f7eadcea4bfe4bf2fe0037f28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5676.56591796875]], [[5663.16357421875]], [[5695.99462890625]], [[5822.51123046875]], [[5712.19482421875]], [[5758.57666015625]], [[5960.83740234375]], [[5803.4951171875]], [[5679.8798828125]], [[5841.8369140625]], [[6021.41064453125]], [[5974.576171875]], [[5505.0576171875]], [[6047.02587890625]], [[5982.21826171875]], [[5856.41943359375]], [[5574.9541015625]], [[5965.71630859375]], [[5789.8740234375]], [[5485.763671875]], [[5671.0224609375]], [[5941.5673828125]], [[5690.8876953125]], [[5785.0322265625]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c93ea88d5206df110d6af1e49de78ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[35946.859375]], [[30698.365234375]], [[34966.9765625]], [[28342.60546875]], [[35975.96875]], [[31918.951171875]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ca57c310677a85c765448f704f4e5452(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54fc3eb66b7b4ff1f00cf7ed8e1c9e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6276.69482421875]], [[6269.55712890625]], [[6866.7685546875]], [[6666.8369140625]], [[6670.61328125]], [[6464.193359375]], [[6803.0732421875]], [[6605.51416015625]], [[6715.45751953125]], [[6639.5166015625]], [[6477.02587890625]], [[6686.94091796875]], [[6197.01513671875]], [[6773.8212890625]], [[6676.494140625]], [[6790.47998046875]], [[6604.2666015625]], [[6252.87744140625]], [[6685.7626953125]], [[6527.548828125]], [[6402.29736328125]], [[6585.9267578125]], [[6778.3955078125]], [[6425.28466796875]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc1d4cad4f908cf2fdc6e9719acfeac1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[32257.3515625]], [[42317.828125]], [[36184.05078125]], [[43102.14453125]], [[41055.6953125]], [[42209.75]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d8db605944e0a3b24c8b52563278faa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a0afca6290e8f323b92cdfb67bad32a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7393.69921875]], [[6959.927734375]], [[6885.21142578125]], [[7081.1806640625]], [[6828.68310546875]], [[6978.8173828125]], [[7253.95556640625]], [[6911.44482421875]], [[6643.69873046875]], [[6981.48681640625]], [[7101.349609375]], [[6820.43701171875]], [[7174.490234375]], [[7039.0732421875]], [[6848.23583984375]], [[7150.19091796875]], [[6990.35400390625]], [[7044.79296875]], [[7003.98486328125]], [[6655.697265625]], [[7098.2880859375]], [[7214.51123046875]], [[6666.53515625]], [[6890.966796875]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b4d0a083eb613d8aa9f121b1ee2f058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[43831.0625]], [[38675.953125]], [[44727.51953125]], [[49158.13671875]], [[41155.8046875]], [[33434.2421875]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7792b31cd5b8ee0ea5a406d230c11d03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3109cb88fb8ace2a51ce19fff42aa2e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6760.47265625]], [[7432.87060546875]], [[7186.91162109375]], [[7512.80029296875]], [[7280.09765625]], [[7307.42626953125]], [[6763.1826171875]], [[6897.64599609375]], [[7159.32275390625]], [[7423.1318359375]], [[7229.619140625]], [[7566.18896484375]], [[7421.4619140625]], [[7363.85546875]], [[7039.63916015625]], [[7469.52490234375]], [[7052.36279296875]], [[7342.88037109375]], [[7058.38134765625]], [[7150.5966796875]], [[7564.6572265625]], [[7208.26611328125]], [[7005.16015625]], [[6895.7578125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b370465d0c19279658607b6009316c7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[38319.57421875]], [[47223.0859375]], [[31635.189453125]], [[44665.33203125]], [[37245.75390625]], [[39227.7265625]]]], dtype='float32').reshape([1, 6, 1, 1]),
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11cd215e3798c4a95747d71e4c183ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([27, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_262d2138d50420e9c8ece0e16b89230e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bc35d26f57dbad91462351d114af4f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c9f6a33db26f36800f5b05525c58a5a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7868a303fac26c9f04f1a3b77fc27739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([240, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9e2eab064255804d0a8334b99e2ead2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([960, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d95b332d11e3cc6e5cb359d91c0a0ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 96, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_49349b6105717e92618bb871250a7115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([72, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c391da67c8299062e7379ab3e8be9bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([288, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce3318ea808029f56edac1ab79709fe9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f377986ce9aed23ad2a17f8d645b8a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([144, 6, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ace2b7e903301abf683245cac4f08ca0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e3e3797444897f0374951d1a6edb160(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1649be3cfdd7ad32ccf3da878a2c444(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5586349df1449ff47eeb1060d89834fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_457b88bb4cb5edd98094c6ad43186388(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.259962558746338]], [[6.480357646942139]], [[6.354485511779785]], [[6.02998685836792]], [[6.0928874015808105]], [[5.784942626953125]], [[5.731517314910889]], [[5.884207725524902]], [[5.937573432922363]], [[5.833317279815674]], [[5.603264808654785]], [[6.116853713989258]], [[6.1093668937683105]], [[6.023929595947266]], [[5.659120082855225]], [[5.759819507598877]], [[5.9197893142700195]], [[6.5027594566345215]], [[5.223503589630127]], [[5.036237716674805]], [[5.466402530670166]], [[6.359437942504883]], [[5.307007312774658]], [[6.603766441345215]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_030e87cd24e6d707149743d672f7f1d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da37eea445df6759ec5737fbd9b8df56
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df9e0866eba4ebeff6965a083b68df76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f8697189f9859721a8a7068ec7d8ab6
    def get_inputs(self):
        return [
            paddle.uniform([11, 3, 224, 224], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 3, 4, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c9774a390c80ba14814b679b14648eb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0710daf91dcdcb297f5868d58f3b77f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1db8806c4e82c27fddd06a15cddec13e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 29], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2fd8a3d614c8692ec9a97d381654d7f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 29], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20538c2d00aba91ac32b0653f200bda7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e8c2cc5b68e73b4b255c87ccbd8c009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1152, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e6ee55823199c82b05551890046e13d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([72, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d520e04d26f949730816e6b231438a85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2b5a260ce0e9f7cc2b53f12695f09884(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([3, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efbdca9d353a0dc70a999c0eab6c6d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([168, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a70f678450ee47c9942ce2efb794697b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([672, 168, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f89b88b29ad4a80db416a636b3201805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4b439aa6664e49919ec72d3ceb3ab52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6d4d6ba0e9a0c57a7cdc5af1e2a1326
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 256, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_762668efbf2f0592b227c2d2fccdc750(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([10, 3840, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1000, 3840, 1, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()