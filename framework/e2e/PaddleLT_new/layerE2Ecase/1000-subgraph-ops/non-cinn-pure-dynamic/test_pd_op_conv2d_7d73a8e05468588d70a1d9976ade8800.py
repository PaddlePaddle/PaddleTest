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


class TestPrimitiveOp_a846c3f1a23e77ab38e5e43aba9d8ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.341363906860352]], [[8.345094680786133]], [[8.33063793182373]], [[8.891763687133789]], [[7.898582935333252]], [[8.031394004821777]], [[7.580002307891846]], [[8.074787139892578]], [[7.623562335968018]], [[8.525352478027344]], [[9.265737533569336]], [[7.251461982727051]], [[7.469519138336182]], [[8.449827194213867]], [[6.704586982727051]], [[7.329336643218994]], [[8.022858619689941]], [[7.848570346832275]], [[8.836841583251953]], [[8.346170425415039]], [[7.800962448120117]], [[8.04752254486084]], [[7.770833969116211]], [[7.5968756675720215]], [[7.5983500480651855]], [[7.384743690490723]], [[7.849663734436035]], [[7.752751350402832]], [[8.344961166381836]], [[8.521525382995605]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_33ee8de256aaa4b98ebaf77dcf5715de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.03987486660480499]], [[0.3108583390712738]], [[0.19754552841186523]], [[0.4267687201499939]], [[0.06886505335569382]], [[0.2773708403110504]], [[0.31698712706565857]], [[0.18207737803459167]], [[0.2122684270143509]], [[0.49441760778427124]], [[0.49583128094673157]], [[0.35014161467552185]], [[0.018081828951835632]], [[0.11968538165092468]], [[0.051485538482666016]], [[0.2947540283203125]], [[0.4430566430091858]], [[0.30342721939086914]], [[0.3534769117832184]]]], dtype='float32').reshape([1, 19, 1, 1]),
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


class TestPrimitiveOp_08732265db93db33489048c33c15c906(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.393464088439941]], [[8.366008758544922]], [[8.483171463012695]], [[8.66169261932373]], [[8.777335166931152]], [[8.07479476928711]], [[8.880349159240723]], [[8.67391300201416]], [[8.541731834411621]], [[7.9030442237854]], [[7.630783557891846]], [[8.261388778686523]], [[8.464116096496582]], [[8.459275245666504]], [[8.572575569152832]], [[8.322819709777832]], [[9.108028411865234]], [[8.179729461669922]], [[8.376543045043945]], [[7.334713459014893]], [[8.867982864379883]], [[8.163381576538086]], [[8.024945259094238]], [[8.204413414001465]], [[8.69442081451416]], [[7.682368755340576]], [[7.935717582702637]], [[8.263702392578125]], [[7.737281322479248]], [[7.696122646331787]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_9c4f4fdf5bcdda53071bc0f284b222a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.025266969576478004]], [[0.1466052085161209]], [[0.24132506549358368]], [[0.34312203526496887]], [[0.33107420802116394]], [[0.4850856065750122]], [[0.08932050317525864]], [[0.000958259857725352]], [[0.2489413172006607]], [[0.36456069350242615]], [[0.036499086767435074]], [[0.246744304895401]], [[0.40913495421409607]], [[0.15042388439178467]], [[0.427543044090271]], [[0.02623920515179634]], [[0.40083417296409607]], [[0.20861923694610596]], [[0.31697648763656616]], [[0.026788970455527306]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([5, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09b87da1b9527bb069b8cbce007d59be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.1495932340621948]], [[1.3737449645996094]], [[1.5405479669570923]], [[1.6220264434814453]], [[1.3335896730422974]]]], dtype='float32').reshape([1, 5, 1, 1]),
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


class TestPrimitiveOp_d4ee276b7a403aae22df2b106ac75afc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.8691492080688477]], [[2.6245875358581543]], [[2.3653178215026855]], [[2.166766405105591]], [[2.307394027709961]], [[2.743351697921753]], [[2.8373217582702637]], [[2.3402910232543945]], [[2.6781842708587646]], [[2.5950942039489746]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_0a3d8d4bee51fc2a1bfe5e45addec536(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.132702350616455]], [[6.780191898345947]], [[5.928580284118652]], [[7.082545757293701]], [[6.498932361602783]], [[6.148091793060303]], [[6.590491771697998]], [[6.749575138092041]], [[6.537431240081787]], [[6.507152557373047]], [[6.074974536895752]], [[6.766306400299072]], [[5.595406532287598]], [[6.529210090637207]], [[7.224152088165283]], [[6.767131328582764]], [[6.367806911468506]], [[6.925047397613525]], [[5.66684627532959]], [[6.8891825675964355]], [[7.272857666015625]], [[7.604793071746826]], [[7.038225173950195]], [[5.851436138153076]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_478a2c4f4d8f3b091bec36469cef697b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.524796962738037]], [[5.10005521774292]], [[4.674687385559082]], [[3.8786232471466064]], [[5.400389671325684]], [[4.489811897277832]], [[5.00974702835083]], [[4.751856803894043]], [[4.238439559936523]], [[4.446267127990723]], [[4.643568515777588]], [[4.354197978973389]], [[4.672003746032715]], [[4.485158443450928]], [[5.109180450439453]], [[4.733465194702148]], [[4.423995494842529]], [[4.591418266296387]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_4d11b233630b1b534d20958eef9a6e27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.493211269378662]], [[6.386541366577148]], [[7.420287609100342]], [[7.333436965942383]], [[6.9355950355529785]], [[5.896524906158447]], [[6.753997325897217]], [[6.382160663604736]], [[6.178173542022705]], [[6.534293174743652]], [[6.505472183227539]], [[6.507557392120361]], [[6.691187858581543]], [[6.243559837341309]], [[6.23176383972168]], [[6.602023124694824]], [[6.351990222930908]], [[6.180388927459717]], [[6.737544536590576]], [[6.839259147644043]], [[7.146553993225098]], [[7.609710693359375]], [[7.2124433517456055]], [[6.125067710876465]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_53659de967c1931b80582574ad7055c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.34306931495666504]], [[0.060790758579969406]], [[0.1307602971792221]], [[0.2543679475784302]], [[0.2366502285003662]], [[0.3382484018802643]], [[0.46558132767677307]], [[0.2893486022949219]], [[0.20847025513648987]], [[0.19089676439762115]], [[0.3397982120513916]], [[0.2802231013774872]], [[0.29074355959892273]], [[0.3012228310108185]], [[0.20100463926792145]], [[0.14693452417850494]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.uniform([4, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1b1ace64f5bb9d206c4423602003abf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0480040311813354]], [[1.208018183708191]], [[1.4790951013565063]], [[1.330366849899292]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


class TestPrimitiveOp_4462a9185560cf9914190a90703f3087(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.2938592433929443]], [[3.408022165298462]], [[3.3406291007995605]], [[3.058248519897461]], [[3.145562171936035]], [[3.171884536743164]], [[3.1627354621887207]], [[3.6159627437591553]], [[2.859837055206299]], [[3.311866044998169]], [[3.52898907661438]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


class TestPrimitiveOp_0b9c7c2848631ac7db7b85d75997a2a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.96597957611084]], [[7.284008979797363]], [[8.071146965026855]], [[8.760475158691406]], [[8.251751899719238]], [[7.766061782836914]], [[8.217127799987793]], [[9.190387725830078]], [[7.507880210876465]], [[9.322604179382324]], [[8.097240447998047]], [[7.6459431648254395]], [[7.849544525146484]], [[8.649650573730469]], [[7.423149108886719]], [[7.311344146728516]], [[7.924724578857422]], [[9.053340911865234]], [[9.133321762084961]], [[8.5658597946167]], [[8.386237144470215]], [[7.7710771560668945]], [[8.541682243347168]], [[7.844338417053223]], [[8.349851608276367]], [[7.915876865386963]], [[8.554839134216309]], [[8.232840538024902]], [[8.90272045135498]], [[8.765819549560547]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_b7c59fb5e752a94d89882561ee6793c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.758634567260742]], [[5.082086563110352]], [[4.171976089477539]], [[3.97188663482666]], [[4.684258937835693]], [[4.020765781402588]], [[4.118278980255127]], [[4.071583271026611]], [[4.342146396636963]], [[3.982689142227173]], [[3.666200637817383]], [[4.092992305755615]], [[4.814263820648193]], [[4.547019958496094]], [[3.568892240524292]], [[3.9223012924194336]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_8910d810d49c3fd85170b24b1645e95d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 7581, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.09964463114738464]], [[0.345501571893692]], [[0.25068527460098267]], [[0.42866596579551697]], [[0.37393295764923096]], [[0.17414391040802002]], [[0.44502440094947815]], [[0.061848234385252]], [[0.16584894061088562]], [[0.05562830716371536]], [[0.40513792634010315]], [[0.029215700924396515]], [[0.4702491760253906]], [[0.12940561771392822]], [[0.19271527230739594]], [[0.39745375514030457]], [[0.23835887014865875]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_19dec5d5d2e3bb8adb88349340e716f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 4725, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.4406029284000397]], [[0.35639357566833496]], [[0.14294719696044922]], [[0.04569276049733162]], [[0.29041996598243713]], [[0.49339592456817627]], [[0.2827490568161011]], [[0.22590602934360504]], [[0.2768259644508362]], [[0.1807122677564621]], [[0.008674395270645618]], [[0.12664146721363068]], [[0.0007052997243590653]], [[0.1933586448431015]], [[0.20864041149616241]], [[0.3819349706172943]], [[0.10404443740844727]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_56bdfb29fdfe9ae4807acf9d5637b27e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.051290988922119]], [[8.006352424621582]], [[8.431422233581543]], [[8.261886596679688]], [[7.675865650177002]], [[7.161308765411377]], [[8.586992263793945]], [[7.49012565612793]], [[6.815402984619141]], [[7.385681629180908]], [[7.945213317871094]], [[7.4884138107299805]], [[7.443033218383789]], [[7.670413017272949]], [[7.075477123260498]], [[7.630590438842773]], [[7.0837931632995605]], [[7.8320512771606445]], [[8.103216171264648]], [[7.677859306335449]], [[7.417149543762207]], [[7.247952938079834]], [[6.757908821105957]], [[7.820258617401123]], [[8.067765235900879]], [[7.257658958435059]], [[9.007942199707031]], [[7.396412372589111]], [[8.738997459411621]], [[6.798262596130371]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_49008c4daf375ae09f6d304ff264b49c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.892675399780273]], [[6.968152046203613]], [[6.0143609046936035]], [[6.4988789558410645]], [[5.598384380340576]], [[5.6200270652771]], [[5.922160625457764]], [[6.1982197761535645]], [[6.238831043243408]], [[6.700442314147949]], [[6.08521842956543]], [[6.50688362121582]], [[6.094415187835693]], [[6.457919597625732]], [[6.121939182281494]], [[6.495532512664795]], [[5.121353626251221]], [[6.021153450012207]], [[6.228803634643555]], [[6.077028751373291]], [[6.203137397766113]], [[5.798994064331055]], [[4.395704746246338]], [[5.763790607452393]], [[6.030169486999512]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_96960b337a058a457fd58d97f245a7fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 8400, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.08194046467542648]], [[0.1043936163187027]], [[0.4631231725215912]], [[0.20496901869773865]], [[0.28380683064460754]], [[0.46008914709091187]], [[0.29294317960739136]], [[0.17166712880134583]], [[0.2267124503850937]], [[0.288276344537735]], [[0.03980851545929909]], [[0.4904789328575134]], [[0.12430858612060547]], [[0.46302351355552673]], [[0.13212572038173676]], [[0.28068122267723083]], [[0.1457175612449646]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_add1211b00874caf6f947319eb99855f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.78770112991333]], [[5.173911094665527]], [[5.828648090362549]], [[5.295206069946289]], [[5.438414096832275]], [[5.914935111999512]], [[5.354312419891357]], [[5.041213512420654]], [[5.047346115112305]], [[5.584328651428223]], [[6.354373455047607]], [[5.3775105476379395]], [[5.281128406524658]], [[5.093140602111816]], [[4.823072910308838]], [[6.270191669464111]], [[5.606038570404053]], [[6.288185119628906]], [[5.714950084686279]], [[5.566715717315674]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_0c6248995061743437cefec9d382d062(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3810640871524811]], [[0.34479349851608276]], [[0.4085673391819]], [[0.1296308934688568]], [[0.2512799799442291]], [[0.1133129745721817]], [[0.4902740716934204]], [[0.38251981139183044]], [[0.4548059403896332]], [[0.4693087637424469]], [[0.35437801480293274]], [[0.3000442385673523]], [[0.11600154638290405]], [[0.293574720621109]], [[0.19377829134464264]], [[0.2614496946334839]], [[0.3989219069480896]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_5699630479b4e0b90afcf033ab542f27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.502219200134277]], [[5.28275203704834]], [[5.1589226722717285]], [[4.98879861831665]], [[5.378580093383789]], [[5.263552188873291]], [[5.741299629211426]], [[5.16142463684082]], [[4.911098480224609]], [[4.91981840133667]], [[5.200071334838867]], [[5.248149871826172]], [[4.898805141448975]], [[5.904212951660156]], [[5.408774375915527]], [[5.2532830238342285]], [[5.063814640045166]], [[4.795100212097168]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_1c4fcf9aa33d981813231038bafc76cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.264429569244385]], [[3.588606834411621]], [[3.901416063308716]], [[4.151577472686768]], [[4.0111918449401855]], [[3.7010557651519775]], [[4.194592475891113]], [[3.618128776550293]], [[4.360840797424316]], [[4.132993221282959]], [[4.101284027099609]], [[4.483236312866211]], [[4.194448947906494]], [[3.754594326019287]], [[3.4575321674346924]], [[4.242446422576904]], [[4.110925197601318]], [[3.6827917098999023]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_1d542121e9d35e9e692e03b514644973(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.562326908111572]], [[5.669094562530518]], [[5.329018592834473]], [[5.497073650360107]], [[5.421993732452393]], [[5.694520950317383]], [[6.503623008728027]], [[5.560434341430664]], [[5.663275241851807]], [[6.299114227294922]], [[5.959976673126221]], [[6.512989044189453]], [[6.62641716003418]], [[5.377169609069824]], [[5.499907493591309]], [[5.698286533355713]], [[5.877714157104492]], [[5.456587791442871]], [[6.124011039733887]], [[6.246454238891602]], [[5.705720901489258]], [[5.565013408660889]], [[5.930296897888184]], [[5.865280628204346]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_beaa04d65b765143fbe624ad4d4512e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.008657455444336]], [[6.322399139404297]], [[5.315286636352539]], [[6.074441909790039]], [[5.413235664367676]], [[5.292750358581543]], [[5.546027183532715]], [[5.273697376251221]], [[5.29599142074585]], [[5.233922004699707]], [[4.632728576660156]], [[4.674685478210449]], [[5.450929164886475]], [[5.20733118057251]], [[5.647550582885742]], [[5.515830039978027]], [[5.297325134277344]], [[5.323426723480225]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_0b66dd2e7ae273051347e5f78203d12d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.00966215133667]], [[4.738468170166016]], [[4.393519878387451]], [[4.130898475646973]], [[4.7210307121276855]], [[5.321023941040039]], [[4.781608581542969]], [[4.954293251037598]], [[5.046009063720703]], [[4.331708908081055]], [[4.6108527183532715]], [[4.21380090713501]], [[4.622811317443848]], [[4.676739692687988]], [[5.098243713378906]], [[4.279331684112549]], [[5.087551593780518]], [[4.904784202575684]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_5d707ec3ab77b9b536baafccccda4cd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 4116, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.485444575548172]], [[0.1007329523563385]], [[0.26445266604423523]], [[0.3921942412853241]], [[0.05905245617032051]], [[0.2954080104827881]], [[0.1132320910692215]], [[0.40070950984954834]], [[0.3168421983718872]], [[0.2777528762817383]], [[0.1389370560646057]], [[0.3621482849121094]], [[0.1746079921722412]], [[0.019861673936247826]], [[0.17883750796318054]], [[0.31226444244384766]], [[0.3710225820541382]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_a7c9854ebf416002955509af40951195(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 6069, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.07810749113559723]], [[0.39910784363746643]], [[0.32437118887901306]], [[0.21621030569076538]], [[0.3104988634586334]], [[0.09112900495529175]], [[0.45704278349876404]], [[0.37612301111221313]], [[0.10071074962615967]], [[0.248264878988266]], [[0.0032806419767439365]], [[0.4789884686470032]], [[0.2242172658443451]], [[0.4822772443294525]], [[0.1559908092021942]], [[0.25957298278808594]], [[0.2791783809661865]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_373a3c3e09de08a0659b4f42d1c1ef48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.570652961730957]], [[5.290533542633057]], [[5.236232757568359]], [[4.753215789794922]], [[5.1592206954956055]], [[5.3395280838012695]], [[4.476436138153076]], [[4.3913350105285645]], [[4.970709323883057]], [[4.858014106750488]], [[5.289392471313477]], [[5.0237956047058105]], [[5.544734477996826]], [[5.3807373046875]], [[4.539039611816406]], [[4.546725749969482]], [[4.030461311340332]], [[5.68776798248291]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_4f17be14b6abb888e90319eccaa18b7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.044375419616699]], [[3.9695029258728027]], [[4.399412631988525]], [[4.351151943206787]], [[3.939140796661377]], [[3.736872434616089]], [[4.160721778869629]], [[3.6401002407073975]], [[3.593968629837036]], [[4.237102031707764]], [[3.89855694770813]], [[3.894055128097534]], [[3.5025439262390137]], [[3.5617258548736572]], [[3.929076671600342]], [[4.469088554382324]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_ffc421c5e36f95265cdbd8f7c45c8fdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.764894485473633]], [[4.012932777404785]], [[4.432371616363525]], [[4.805740833282471]], [[4.5551652908325195]], [[4.929264545440674]], [[4.797806262969971]], [[5.429898262023926]], [[4.413793087005615]], [[4.965513706207275]], [[5.158843517303467]], [[4.463954448699951]], [[4.748885154724121]], [[4.852672576904297]], [[5.104424476623535]], [[4.424574375152588]], [[5.957918167114258]], [[4.536571502685547]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_4b7ef45e7c9cb66d9ed4a97b97895fcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.4132128655910492]], [[0.2538280487060547]], [[0.20738531649112701]], [[0.08074744045734406]], [[0.39935845136642456]], [[0.2538260519504547]], [[0.44008761644363403]], [[0.10762587934732437]], [[0.4161072075366974]], [[0.3235967457294464]], [[0.374724417924881]], [[0.19214914739131927]], [[0.34028029441833496]], [[0.46019548177719116]], [[0.19836606085300446]], [[0.054474394768476486]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.uniform([4, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82f08b5fbbad24b7d57f337c92c93ab9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.5847580432891846]], [[1.0072780847549438]], [[1.3233182430267334]], [[1.5454227924346924]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


class TestPrimitiveOp_360eba518b563e3cd2079069e8d7c362(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.271772384643555]], [[5.875980377197266]], [[5.233264923095703]], [[5.741209030151367]], [[5.142308712005615]], [[4.55288553237915]], [[5.253786087036133]], [[5.284027099609375]], [[4.941991806030273]], [[5.884860992431641]], [[5.917901992797852]], [[5.081498146057129]], [[5.755356788635254]], [[5.243422031402588]], [[5.570818901062012]], [[4.73895788192749]], [[5.321943759918213]], [[5.601650238037109]], [[5.280106067657471]], [[5.202432155609131]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_3025a3ab24b68b7f34df3ab3e9b89a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.4511351585388184]], [[3.596243381500244]], [[3.079944610595703]], [[3.0792460441589355]], [[3.197981834411621]], [[3.155919313430786]], [[2.571645975112915]], [[3.180319309234619]], [[3.193372964859009]], [[3.010362386703491]], [[2.657683849334717]], [[3.003628969192505]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


class TestPrimitiveOp_887af7814e89d22ec679c6f9a7304f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.712578773498535]], [[4.405738830566406]], [[5.340959548950195]], [[5.020174980163574]], [[5.04586935043335]], [[5.332529544830322]], [[5.075929164886475]], [[5.159215450286865]], [[4.925020694732666]], [[4.820794105529785]], [[4.712738513946533]], [[5.557010650634766]], [[4.693427085876465]], [[5.183993339538574]], [[4.675652980804443]], [[4.937038898468018]], [[5.127405643463135]], [[5.4516282081604]], [[5.081034183502197]], [[5.154932022094727]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_01dfea45e4df347adecd0232a16ae1b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.3770663738250732]], [[3.018279552459717]], [[3.1935417652130127]], [[3.3590152263641357]], [[3.3767566680908203]], [[3.033186197280884]], [[3.3185150623321533]], [[3.309736728668213]], [[3.105422019958496]], [[3.1335480213165283]], [[2.7805066108703613]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


class TestPrimitiveOp_e8caf5c6e5c9405088dfe2d570f934d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.7868828773498535]], [[4.309795379638672]], [[3.672903060913086]], [[3.462696075439453]], [[4.04031229019165]], [[3.203916072845459]], [[4.003729820251465]], [[3.761460065841675]], [[3.279228687286377]], [[3.416935443878174]], [[3.2144813537597656]], [[3.2578201293945312]], [[3.7585110664367676]], [[3.6805272102355957]]]], dtype='float32').reshape([1, 14, 1, 1]),
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


class TestPrimitiveOp_0d9dc576e40a1594e929dd53aaf54dea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.974609375]], [[4.976262092590332]], [[4.851878643035889]], [[4.950084209442139]], [[5.452333450317383]], [[4.530028820037842]], [[5.880236625671387]], [[5.714767932891846]], [[5.143693923950195]], [[4.772283554077148]], [[5.391656875610352]], [[5.164602756500244]], [[5.560387134552002]], [[5.347257137298584]], [[6.138763904571533]], [[5.2376790046691895]], [[5.05681037902832]], [[5.90244197845459]], [[5.363766193389893]], [[5.793654918670654]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_21ef3af48e8d1197a8e5c8bce75eb1e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[30611.162109375]], [[34098.68359375]], [[33542.65234375]], [[37874.97265625]], [[35785.0390625]], [[30797.640625]]], [[[29474.904296875]], [[32840.2265625]], [[32297.748046875]], [[36470.22265625]], [[34456.5078125]], [[29652.517578125]]]], dtype='float32').reshape([2, 6, 1, 1]),
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


class TestPrimitiveOp_7f334a1394982e663a984023464ddb4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[43153.4375]], [[39524.2578125]], [[34366.5703125]], [[27617.447265625]], [[39732.4453125]], [[34258.45703125]]], [[[41476.13671875]], [[37983.9921875]], [[33026.08984375]], [[26542.349609375]], [[38184.8671875]], [[32926.66796875]]]], dtype='float32').reshape([2, 6, 1, 1]),
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


class TestPrimitiveOp_7a1c9c4cd73e2d4c82e094a7b5b3f706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[46980.84375]], [[49469.58984375]], [[34778.47265625]], [[41731.0703125]], [[44243.390625]], [[45118.13671875]]], [[[44473.92578125]], [[46829.0859375]], [[32919.1796875]], [[39499.72265625]], [[41876.26171875]], [[42702.77734375]]]], dtype='float32').reshape([2, 6, 1, 1]),
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


class TestPrimitiveOp_cfbf786a575bbb98a12841878ee9839e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[48133.4765625]], [[41553.1015625]], [[39663.78515625]], [[38959.29296875]], [[47156.734375]], [[43251.9921875]]], [[[45779.3359375]], [[39522.046875]], [[37720.71875]], [[37059.18359375]], [[44854.62890625]], [[41139.7734375]]]], dtype='float32').reshape([2, 6, 1, 1]),
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


class TestPrimitiveOp_943ea444fcb6b32ce9793aac84e924ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 9261, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2801448106765747]], [[0.30296146869659424]], [[0.13105881214141846]], [[0.2884991765022278]], [[0.3615635335445404]], [[0.3544272482395172]], [[0.05501127243041992]], [[0.2766587436199188]], [[0.001432210672646761]], [[0.3616071939468384]], [[0.15316426753997803]], [[0.48606011271476746]], [[0.3304632008075714]], [[0.30052274465560913]], [[0.11840054392814636]], [[0.17324790358543396]], [[0.48850327730178833]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_17a8584421456059175ae2a41a710227(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.534451484680176]], [[7.903776168823242]], [[9.045938491821289]], [[7.593854904174805]], [[7.122854709625244]], [[8.03244686126709]], [[7.656867027282715]], [[7.268033027648926]], [[7.397475719451904]], [[7.093584060668945]], [[8.421623229980469]], [[8.075343132019043]], [[7.709201812744141]], [[7.9542317390441895]], [[7.758068084716797]], [[7.648263931274414]], [[6.878692150115967]], [[8.587495803833008]], [[8.04971981048584]], [[6.730032444000244]], [[7.359399318695068]], [[8.338031768798828]], [[7.228526592254639]], [[8.696207046508789]], [[7.203897476196289]], [[6.891290664672852]], [[7.43543004989624]], [[7.630880832672119]], [[7.928101062774658]], [[7.90075159072876]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_d556bc65923b1da6a14b3e5b3d24f03a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.7160964012146]], [[8.8593111038208]], [[7.149313926696777]], [[7.997802734375]], [[7.889408588409424]], [[8.159114837646484]], [[8.61439323425293]], [[8.108988761901855]], [[7.816248893737793]], [[8.484747886657715]], [[7.5950236320495605]], [[8.478062629699707]], [[7.839389324188232]], [[8.435235977172852]], [[7.654603958129883]], [[7.260209083557129]], [[7.277437210083008]], [[7.615996837615967]], [[8.556742668151855]], [[8.04092788696289]], [[7.731863498687744]], [[7.525420665740967]], [[8.144274711608887]], [[8.088244438171387]], [[8.052762985229492]], [[8.022008895874023]], [[8.075109481811523]], [[8.527801513671875]], [[7.209123134613037]], [[7.500683307647705]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_ee0c68d3700392bf6be618d23641ece4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.071221351623535]], [[8.483712196350098]], [[7.512153625488281]], [[7.391823768615723]], [[7.566353797912598]], [[7.654898643493652]], [[8.039820671081543]], [[8.686598777770996]], [[8.012916564941406]], [[6.911598205566406]], [[8.386569023132324]], [[8.96761417388916]], [[8.04858112335205]], [[7.871091842651367]], [[7.638219833374023]], [[7.861978054046631]], [[8.267280578613281]], [[7.594769477844238]], [[9.33325481414795]], [[8.84327507019043]], [[7.729835033416748]], [[8.22775650024414]], [[8.440366744995117]], [[7.035523891448975]], [[8.408729553222656]], [[7.929727077484131]], [[7.558557033538818]], [[8.91915512084961]], [[7.66169548034668]], [[8.36217212677002]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_1f9393071060eedb01ae5d2826f2a7b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.112114906311035]], [[8.157021522521973]], [[8.03063678741455]], [[7.852174758911133]], [[8.124262809753418]], [[8.385584831237793]], [[7.801120758056641]], [[7.293982028961182]], [[7.800896644592285]], [[8.04080581665039]], [[8.630279541015625]], [[7.310001373291016]], [[7.316909313201904]], [[8.372149467468262]], [[7.76572322845459]], [[8.136466979980469]], [[7.5545573234558105]], [[7.791067123413086]], [[7.560868263244629]], [[7.211294651031494]], [[7.960395812988281]], [[7.222629547119141]], [[7.690578937530518]], [[7.012873649597168]], [[8.222583770751953]], [[7.545907020568848]], [[7.361056327819824]], [[7.651693820953369]], [[8.053607940673828]], [[8.189615249633789]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_f1710cb24fb6d008c1520b33ea21973d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.5401933193206787]], [[3.6423428058624268]], [[3.0797853469848633]], [[3.44569730758667]], [[3.7470006942749023]], [[3.3460614681243896]], [[3.2297539710998535]], [[3.406916618347168]], [[3.404538154602051]], [[3.162586212158203]], [[3.4384658336639404]], [[2.78239369392395]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


class TestPrimitiveOp_1e74e77ce793fbafff88c573b1a310cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.8037774562835693]], [[3.8427181243896484]], [[3.481290578842163]], [[3.867964029312134]], [[3.9030420780181885]], [[3.2974624633789062]], [[3.8915672302246094]], [[3.6613614559173584]], [[3.5692715644836426]], [[3.7305221557617188]], [[3.3878841400146484]], [[3.550487518310547]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


class TestPrimitiveOp_1e9eb043dd67590b773b18e434a53e79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.2766432762146]], [[6.426517009735107]], [[6.425320148468018]], [[6.268661022186279]], [[6.211418628692627]], [[6.589906692504883]], [[6.614333152770996]], [[6.040896415710449]], [[6.378408908843994]], [[5.200664520263672]], [[5.978342533111572]], [[6.291723251342773]], [[6.274717330932617]], [[6.295009136199951]], [[6.354930400848389]], [[7.007253170013428]], [[5.763204574584961]], [[6.385103702545166]], [[6.056008815765381]], [[6.913020133972168]], [[6.038397312164307]], [[6.664734840393066]], [[7.555033206939697]], [[5.833787441253662]], [[5.681970596313477]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_71adfedb758b9c8e8783a92fb4b45a33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 2100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.08148424327373505]], [[0.2909559905529022]], [[0.0645793080329895]], [[0.10351197421550751]], [[0.3156776428222656]], [[0.13204292953014374]], [[0.32790711522102356]], [[0.46448108553886414]], [[0.38398128747940063]], [[0.36913028359413147]], [[0.4458972215652466]], [[0.39839404821395874]], [[0.25091469287872314]], [[0.25582262873649597]], [[0.05251312628388405]], [[0.0003716308274306357]], [[0.3369043171405792]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_fca35887e8a7d24a86ae316032c7d7ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.109279632568359]], [[5.014555931091309]], [[4.3691840171813965]], [[4.839135646820068]], [[4.961926460266113]], [[4.7770094871521]], [[4.772217273712158]], [[5.030905723571777]], [[4.831387996673584]], [[4.453762531280518]], [[4.867634296417236]], [[4.604258060455322]], [[4.665245056152344]], [[5.421208381652832]], [[4.70947265625]], [[4.673716068267822]], [[5.502685546875]], [[4.979745388031006]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_81147bf293b6c1efdd8fd2336c17d942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.033360887318849564]], [[0.4547981917858124]], [[0.451895534992218]], [[0.21287021040916443]], [[0.49563419818878174]], [[0.4175942540168762]], [[0.38323071599006653]], [[0.39565157890319824]], [[0.4337523877620697]], [[0.43257826566696167]], [[0.03635707125067711]], [[0.06410732120275497]], [[0.08872469514608383]], [[0.12978972494602203]], [[0.18203800916671753]], [[0.05429258942604065]], [[0.31077897548675537]], [[0.23423722386360168]], [[0.3925115168094635]], [[0.43541643023490906]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([5, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9bb27fa0eb4163faa7d738afc01b7398(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.5986579656600952]], [[2.3876352310180664]], [[1.7807573080062866]], [[1.909144401550293]], [[2.1521639823913574]]]], dtype='float32').reshape([1, 5, 1, 1]),
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


class TestPrimitiveOp_130102cf381a4dfc4a80793dd171d281(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.8739213943481445]], [[2.9864583015441895]], [[2.4922564029693604]], [[3.1521081924438477]], [[2.8864846229553223]], [[2.758599281311035]], [[3.44002103805542]], [[3.126253843307495]], [[3.232464551925659]], [[2.122493267059326]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_e27d04da05d0cad263cecaa22ccec503(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.8211588859558105]], [[5.969562530517578]], [[5.040027618408203]], [[5.545391082763672]], [[5.9392828941345215]], [[5.698614597320557]], [[6.020728588104248]], [[5.518492221832275]], [[5.775483131408691]], [[5.343318462371826]], [[6.327739715576172]], [[6.019591808319092]], [[5.764501571655273]], [[6.030677795410156]], [[4.94651460647583]], [[5.521027565002441]], [[5.575688362121582]], [[6.397119045257568]], [[5.447817802429199]], [[6.309340476989746]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_a8088367927caba51530911f5725c6e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 11109, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.32842710614204407]], [[0.4748085141181946]], [[0.38206183910369873]], [[0.41218069195747375]], [[0.050998788326978683]], [[0.1348484605550766]], [[0.15068447589874268]], [[0.10689844191074371]], [[0.45550182461738586]], [[0.04318750649690628]], [[0.14791284501552582]], [[0.2595992386341095]], [[0.1596067100763321]], [[0.4677114188671112]], [[0.31338438391685486]], [[0.29288116097450256]], [[0.08191602677106857]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_7331af195b286d4a22a0e765a0eeb6a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.330204010009766]], [[7.411655902862549]], [[7.0330491065979]], [[6.3860554695129395]], [[6.124695777893066]], [[7.14809513092041]], [[7.321486473083496]], [[6.776966571807861]], [[6.833548545837402]], [[7.155159950256348]], [[7.691615104675293]], [[7.112806797027588]], [[7.824557304382324]], [[6.744721412658691]], [[6.807538032531738]], [[6.391137599945068]], [[6.088601112365723]], [[6.99222469329834]], [[6.901208877563477]], [[7.331323623657227]], [[7.388026237487793]], [[6.604344367980957]], [[7.211657524108887]], [[6.921568870544434]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_e522a9ad747e3f02a236ba1560b7efc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.089078426361084]], [[2.8663032054901123]], [[2.343900203704834]], [[2.262293815612793]], [[2.3554248809814453]], [[2.629438638687134]], [[2.2276692390441895]], [[2.408189058303833]], [[2.5986056327819824]], [[2.2128899097442627]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_9e1db904f49f3c2770af3b37e7d97f07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.703721523284912]], [[4.271613121032715]], [[4.065333843231201]], [[3.9800946712493896]], [[4.790715217590332]], [[4.447109222412109]], [[4.510074615478516]], [[4.803034782409668]], [[4.372105121612549]], [[4.575693130493164]], [[3.1528213024139404]], [[4.1718058586120605]], [[4.4475884437561035]], [[4.449507713317871]], [[4.6612372398376465]], [[4.065131664276123]], [[4.700448036193848]], [[4.4108357429504395]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_5abe6681cbcff23876e41409ed42aecb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.3440985679626465]], [[7.084399700164795]], [[7.289212226867676]], [[6.920706748962402]], [[8.188225746154785]], [[8.090666770935059]], [[6.819272518157959]], [[7.7566986083984375]], [[7.840989112854004]], [[7.760538101196289]], [[7.646232604980469]], [[7.296329021453857]], [[6.769723415374756]], [[6.9615092277526855]], [[8.337942123413086]], [[7.496492385864258]], [[7.324259281158447]], [[7.929457187652588]], [[8.382543563842773]], [[7.682465553283691]], [[8.053009986877441]], [[6.8763203620910645]], [[6.737455368041992]], [[7.033125877380371]], [[7.677404880523682]], [[7.813915729522705]], [[7.640761375427246]], [[7.280036449432373]], [[7.833850860595703]], [[7.040121078491211]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_122d633a9fd70f7e75fd42a7f8197a82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.2524843215942383]], [[0.10555963218212128]], [[0.31403228640556335]], [[0.1089850515127182]], [[0.3079727590084076]], [[0.22917626798152924]], [[0.36001497507095337]], [[0.4083234667778015]], [[0.21860329806804657]], [[0.4403923749923706]], [[0.2974814176559448]], [[0.2387513667345047]], [[0.32789990305900574]], [[0.22906038165092468]], [[0.10392027348279953]], [[0.1165410503745079]], [[0.25199538469314575]], [[0.4455839991569519]], [[0.44833946228027344]], [[0.09627983719110489]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([5, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ccf6395dad0fa6271a0192340ce38caa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.9913126230239868]], [[1.1797109842300415]], [[1.1726734638214111]], [[1.3191273212432861]], [[1.9014294147491455]]]], dtype='float32').reshape([1, 5, 1, 1]),
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


class TestPrimitiveOp_bc6a2142ec49f295619ddf360d8265b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.683842658996582]], [[2.6842782497406006]], [[2.5433034896850586]], [[2.1871120929718018]], [[2.8770945072174072]], [[2.4621968269348145]], [[2.7253870964050293]], [[2.209808349609375]], [[2.8196940422058105]], [[1.9530487060546875]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_a4dc7a72c437c43c10ed96ad6c751363(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.309731960296631]], [[6.070625305175781]], [[5.621429443359375]], [[5.663862705230713]], [[6.513097763061523]], [[5.720493316650391]], [[5.386129856109619]], [[6.173917293548584]], [[6.846288681030273]], [[6.566182613372803]], [[4.968994140625]], [[6.637065887451172]], [[5.679732322692871]], [[5.365406036376953]], [[6.678298473358154]], [[5.641069412231445]], [[5.841551303863525]], [[6.354818344116211]], [[5.365055084228516]], [[5.7638397216796875]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_de5aa6c8e37d0a3aa0f8bd078103af8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.353776454925537]], [[5.755500316619873]], [[4.986583709716797]], [[4.325551986694336]], [[4.293920040130615]], [[5.164376735687256]], [[4.2319016456604]], [[5.23312520980835]], [[5.250441074371338]], [[4.849178314208984]], [[4.646942138671875]], [[4.615958213806152]], [[4.809027671813965]], [[5.336738109588623]], [[4.309662342071533]], [[4.57535457611084]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_a6e458d3748598ee690a8a4b27f94fc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.822312116622925]], [[3.8225741386413574]], [[3.9365592002868652]], [[4.422060966491699]], [[4.283381938934326]], [[4.322755813598633]], [[3.9440715312957764]], [[4.428144931793213]], [[3.9092602729797363]], [[4.2019147872924805]], [[3.6413607597351074]], [[4.337952136993408]], [[4.19197940826416]], [[4.296870708465576]]]], dtype='float32').reshape([1, 14, 1, 1]),
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


class TestPrimitiveOp_b6f1c133ff56a0413ea9084abdb5d913(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.052672386169434]], [[5.049968242645264]], [[4.773495674133301]], [[4.635282039642334]], [[4.586019992828369]], [[4.91532564163208]], [[5.271136283874512]], [[4.549180507659912]], [[5.105238914489746]], [[4.596596717834473]], [[5.254343509674072]], [[4.510046482086182]], [[4.815375328063965]], [[5.0578932762146]], [[4.837278366088867]], [[4.914398670196533]], [[4.987178325653076]], [[5.231625080108643]], [[4.814979076385498]], [[5.43070650100708]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_f2d2fa87ae345a90f652dbce1ac0d0e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.094910621643066]], [[8.318694114685059]], [[8.528326034545898]], [[8.8722562789917]], [[8.198942184448242]], [[7.374891757965088]], [[8.65483570098877]], [[7.33724308013916]], [[8.711739540100098]], [[8.610730171203613]], [[8.831395149230957]], [[7.739767551422119]], [[9.076775550842285]], [[7.7435150146484375]], [[7.959839820861816]], [[7.265451431274414]], [[8.023520469665527]], [[8.827188491821289]], [[8.824771881103516]], [[8.233180046081543]], [[8.71473217010498]], [[8.02358341217041]], [[7.9135823249816895]], [[7.4622650146484375]], [[8.667437553405762]], [[8.51819133758545]], [[7.9772047996521]], [[9.0551118850708]], [[8.486502647399902]], [[8.540457725524902]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_641ab63daa83204b8b7ed01fced60dd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 3024, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.21155066788196564]], [[0.37269607186317444]], [[0.45510122179985046]], [[0.3991148769855499]], [[0.22521258890628815]], [[0.35598236322402954]], [[0.101817287504673]], [[0.20658525824546814]], [[0.283997118473053]], [[0.4088667631149292]], [[0.24974146485328674]], [[0.3156215250492096]], [[0.2862697243690491]], [[0.08065810799598694]], [[0.3317941427230835]], [[0.2112303376197815]], [[0.4414900839328766]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_7f7d97d213b2463c83f0d4b0c2a9f06e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.923881530761719]], [[7.498033046722412]], [[6.0636749267578125]], [[7.236250877380371]], [[6.766568660736084]], [[5.967278957366943]], [[6.8577351570129395]], [[6.778499126434326]], [[6.799393653869629]], [[5.967048168182373]], [[7.128371715545654]], [[6.49347448348999]], [[6.275139808654785]], [[7.331699371337891]], [[6.719854831695557]], [[7.174748420715332]], [[7.164754867553711]], [[6.690661430358887]], [[5.764800548553467]], [[7.2003350257873535]], [[6.125292778015137]], [[6.216479778289795]], [[6.546446800231934]], [[6.464174270629883]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_cdddb1999b037aad144131b80806e45d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.346798896789551]], [[6.081398010253906]], [[7.299726963043213]], [[6.957866668701172]], [[6.698203086853027]], [[6.293685436248779]], [[7.346051216125488]], [[6.788509368896484]], [[6.625288009643555]], [[6.3033623695373535]], [[6.017172813415527]], [[6.440759658813477]], [[6.5812859535217285]], [[6.249759197235107]], [[6.78050422668457]], [[6.303239345550537]], [[5.982887268066406]], [[6.647152900695801]], [[6.8246073722839355]], [[6.154778480529785]], [[6.4065752029418945]], [[7.145350456237793]], [[6.706385612487793]], [[6.898695468902588]], [[6.413028717041016]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_99bf1fb87ad842993b27497709d25093(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.007822036743164]], [[2.990636110305786]], [[2.9978952407836914]], [[3.345715045928955]], [[3.1105501651763916]], [[2.689404249191284]], [[3.436768054962158]], [[2.952461004257202]], [[2.769449234008789]], [[2.8575563430786133]], [[2.4953866004943848]], [[2.7217109203338623]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


class TestPrimitiveOp_9518f7ac0bc29978b16d1bd73c24ff10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[689.8653564453125]], [[680.5704956054688]], [[664.4366455078125]], [[699.2727661132812]], [[696.021240234375]], [[744.6607055664062]], [[655.9004516601562]], [[664.6710815429688]], [[756.8004150390625]], [[715.6314697265625]], [[712.28564453125]], [[669.6095581054688]], [[680.5859375]], [[706.99365234375]], [[732.1513671875]], [[648.2433471679688]], [[719.170166015625]], [[684.26806640625]], [[769.2687377929688]], [[721.9342651367188]], [[705.001220703125]], [[705.870849609375]], [[745.766357421875]], [[657.6587524414062]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_6fcdadf00a333c4b223494f06c43dac4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[77.98493194580078]], [[70.32717895507812]], [[63.509315490722656]], [[75.7135238647461]], [[63.36880111694336]], [[71.58040618896484]], [[70.64983367919922]], [[77.32463073730469]], [[65.07374572753906]], [[73.18685150146484]], [[79.83320617675781]], [[73.63390350341797]], [[78.8461685180664]], [[71.25952911376953]], [[74.20136260986328]], [[76.1542739868164]], [[66.89875030517578]], [[77.56700897216797]], [[71.27639770507812]], [[77.64783477783203]], [[76.48995971679688]], [[71.31640625]], [[75.41777801513672]], [[72.67097473144531]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_a755823d2750a130af188333cf73845d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[34.7319450378418]], [[33.58924102783203]], [[36.77731704711914]], [[32.51150894165039]], [[32.94678497314453]], [[35.561302185058594]], [[30.16822624206543]], [[35.28459548950195]], [[35.693115234375]], [[32.90860366821289]], [[34.64813995361328]], [[34.00236129760742]], [[37.37583541870117]], [[34.15388107299805]], [[36.022220611572266]], [[35.51877212524414]], [[32.52545928955078]], [[34.947906494140625]], [[28.022186279296875]], [[37.637874603271484]], [[35.4067268371582]], [[35.718544006347656]], [[35.86259841918945]], [[34.50102996826172]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_14accaa124ae90ce33104e2058ed4402(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[24.370454788208008]], [[22.815378189086914]], [[24.97918701171875]], [[25.29088592529297]], [[20.115402221679688]], [[24.850099563598633]], [[22.514122009277344]], [[25.646230697631836]], [[23.390384674072266]], [[25.469881057739258]], [[25.39321517944336]], [[23.23244857788086]], [[28.12644386291504]], [[20.19763946533203]], [[22.85979652404785]], [[24.788991928100586]], [[25.283416748046875]], [[26.703161239624023]], [[23.287612915039062]], [[25.041770935058594]], [[24.598798751831055]], [[25.458181381225586]], [[22.87262725830078]], [[24.76409339904785]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_d22bf1f67f5b5bbd6ce0a060a2547dc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5854.14404296875]], [[5659.60205078125]], [[5366.99853515625]], [[5410.21240234375]], [[5898.18798828125]], [[5650.875]], [[5791.6689453125]], [[5669.76171875]], [[5643.39013671875]], [[5588.5712890625]], [[5637.26708984375]], [[5531.98828125]], [[5620.38134765625]], [[5946.662109375]], [[5659.94482421875]], [[5417.81591796875]], [[5317.71533203125]], [[5833.75390625]], [[5530.11279296875]], [[5339.79296875]], [[5585.6220703125]], [[5524.48046875]], [[5693.1083984375]], [[5513.64111328125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f402a7820a9f035b9e5f95276402f1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[32749.275390625]], [[39411.7109375]], [[35941.9140625]], [[26188.5]], [[36667.39453125]], [[34495.625]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_36d309d1c16314981e6ed6329f5923d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6431.806640625]], [[6265.28271484375]], [[6499.75048828125]], [[6331.28564453125]], [[6107.74609375]], [[6474.2646484375]], [[5873.10205078125]], [[6080.80908203125]], [[6254.48681640625]], [[6277.087890625]], [[6418.48291015625]], [[6270.78076171875]], [[6327.84228515625]], [[6336.34375]], [[6297.40869140625]], [[6535.45556640625]], [[6251.0771484375]], [[6278.0009765625]], [[6313.2509765625]], [[6571.1669921875]], [[6390.24267578125]], [[6414.48828125]], [[6466.73974609375]], [[6325.3076171875]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b87dc070f5ff3d6e26308d740f7b62e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[33976.58984375]], [[38068.30078125]], [[37686.6484375]], [[31999.083984375]], [[40803.84765625]], [[39620.8984375]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_81122a4098794ac668b8a13f3f1a78dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6479.41259765625]], [[6311.61279296875]], [[6578.8701171875]], [[6484.68798828125]], [[6347.94677734375]], [[6632.384765625]], [[6488.7490234375]], [[6561.47216796875]], [[6609.35107421875]], [[6515.72412109375]], [[6174.31005859375]], [[6195.134765625]], [[6534.1357421875]], [[6730.29345703125]], [[6498.17724609375]], [[6310.8505859375]], [[6720.18310546875]], [[6650.6875]], [[6653.91015625]], [[6318.6513671875]], [[6334.12890625]], [[6891.18115234375]], [[6357.97900390625]], [[6442.28271484375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a414122ded7602480d2f9739c079134(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[32314.748046875]], [[32420.060546875]], [[44085.0625]], [[39596.10546875]], [[40085.6328125]], [[33267.4453125]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_a7d39522573bd2037b00128338688ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6648.56396484375]], [[6455.53662109375]], [[6643.4248046875]], [[6828.20068359375]], [[6411.69970703125]], [[6836.2470703125]], [[6579.6904296875]], [[6741.75732421875]], [[6969.17138671875]], [[6883.55517578125]], [[6441.4931640625]], [[6669.97265625]], [[6920.759765625]], [[6606.5078125]], [[6803.54345703125]], [[6666.69482421875]], [[6968.1943359375]], [[6707.7734375]], [[7030.3828125]], [[6298.8466796875]], [[6626.8349609375]], [[6381.802734375]], [[6849.36669921875]], [[6894.94091796875]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed39f9885a926317819f763a08703fb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[38635.25390625]], [[39958.2734375]], [[49901.5546875]], [[48418.13671875]], [[44376.90234375]], [[38225.046875]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_0138a4c8b03c3e0c633b76b0f82d9912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.371476173400879]], [[6.853559970855713]], [[6.055513858795166]], [[5.596645832061768]], [[6.152677536010742]], [[6.237435340881348]], [[6.416729927062988]], [[5.967062950134277]], [[6.073554992675781]], [[6.291712284088135]], [[6.719925880432129]], [[5.367959976196289]], [[7.373897552490234]], [[5.144876003265381]], [[6.575024127960205]], [[5.4149909019470215]], [[5.995121955871582]], [[6.610832691192627]], [[5.161527633666992]], [[6.301140785217285]], [[6.229990482330322]], [[5.891112804412842]], [[5.975530624389648]], [[5.764140605926514]]]], dtype='float32').reshape([1, 24, 1, 1]),
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