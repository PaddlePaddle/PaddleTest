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


class TestPrimitiveOp_0e00fc4b53f9ffbfe424f2ebe772226f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.850220203399658]], [[6.410380840301514]], [[7.415620803833008]], [[7.72133731842041]], [[7.379231929779053]], [[7.465376377105713]], [[7.83229923248291]], [[7.3430914878845215]], [[8.433938980102539]], [[8.384262084960938]], [[7.256677627563477]], [[8.074450492858887]], [[8.050472259521484]], [[7.776950836181641]], [[6.175331115722656]], [[7.317542552947998]], [[8.276723861694336]], [[7.710753440856934]], [[6.962830066680908]], [[7.7543535232543945]], [[7.796797275543213]], [[8.071995735168457]], [[7.5873613357543945]], [[7.624943256378174]], [[7.499958038330078]], [[7.734288215637207]], [[8.8361234664917]], [[7.007436752319336]], [[8.493032455444336]], [[8.418644905090332]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_32a8415da48585443bf5b5a8a42bcaf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.07194972783327103]], [[0.3280642330646515]], [[0.20783749222755432]], [[0.3097696006298065]], [[0.03462572395801544]], [[0.23802152276039124]], [[0.22776855528354645]], [[0.07152417302131653]], [[0.1993035078048706]], [[0.44579383730888367]], [[0.08543410152196884]], [[0.47711309790611267]], [[0.2584003210067749]], [[0.4323287010192871]], [[0.019899992272257805]], [[0.402402400970459]], [[0.14829663932323456]], [[0.02997129037976265]], [[0.3538842499256134]]]], dtype='float32').reshape([1, 19, 1, 1]),
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


class TestPrimitiveOp_f6119946397ba9f008f2b90e311140a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.828388690948486]], [[7.152416706085205]], [[7.548887252807617]], [[7.925854682922363]], [[8.590157508850098]], [[7.57591438293457]], [[7.858030319213867]], [[7.5591630935668945]], [[7.303663730621338]], [[7.477190971374512]], [[8.684175491333008]], [[7.87654972076416]], [[7.411413669586182]], [[7.940890312194824]], [[7.757748603820801]], [[8.416932106018066]], [[6.928280353546143]], [[7.116427421569824]], [[8.085753440856934]], [[7.5065717697143555]], [[7.985809803009033]], [[7.27017068862915]], [[7.539728164672852]], [[8.036003112792969]], [[8.134060859680176]], [[7.229857444763184]], [[7.594155788421631]], [[9.057613372802734]], [[8.551019668579102]], [[8.29045581817627]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_df10c6467102450c94b1154e5dc6db09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.44429129362106323]], [[0.056593768298625946]], [[0.39657852053642273]], [[0.33796826004981995]], [[0.49706417322158813]], [[0.3572792410850525]], [[0.03654392436146736]], [[0.21142825484275818]], [[0.40960678458213806]], [[0.43277841806411743]], [[0.2908302843570709]], [[0.14194461703300476]], [[0.360965371131897]], [[0.010443153791129589]], [[0.3076803982257843]], [[0.13433098793029785]], [[0.20657916367053986]], [[0.47308850288391113]], [[0.36641454696655273]], [[0.08617551624774933]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([5, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a91cdee282df47cb6810e751da1e782(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.694800853729248]], [[1.6033475399017334]], [[1.4151287078857422]], [[1.6911625862121582]], [[1.7419650554656982]]]], dtype='float32').reshape([1, 5, 1, 1]),
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


class TestPrimitiveOp_1e3c7cd4e38fb9f7ee35b3ef29d3e705(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.871072292327881]], [[3.109473466873169]], [[3.7252604961395264]], [[2.9413692951202393]], [[2.8883273601531982]], [[2.965113401412964]], [[2.537015914916992]], [[2.602930784225464]], [[2.682055711746216]], [[3.434065818786621]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_bc78c46b71df1ca95920ccb53660e486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.083743572235107]], [[6.656046390533447]], [[6.4518141746521]], [[6.063625812530518]], [[5.930395603179932]], [[6.21452522277832]], [[6.806361198425293]], [[6.955392837524414]], [[6.590692043304443]], [[6.444944381713867]], [[6.871044635772705]], [[6.525993347167969]], [[6.97701358795166]], [[6.446276664733887]], [[6.60465145111084]], [[6.996643543243408]], [[6.897172451019287]], [[5.9756293296813965]], [[7.418559551239014]], [[6.6775898933410645]], [[6.010356426239014]], [[6.246400356292725]], [[6.74877405166626]], [[6.817819118499756]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_2460c15ca7b27eef8bc296b169c0db16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.866799831390381]], [[4.336925029754639]], [[4.4317803382873535]], [[5.035212993621826]], [[5.0111470222473145]], [[4.435763835906982]], [[4.339719295501709]], [[4.078125476837158]], [[4.702528953552246]], [[4.914383888244629]], [[5.69053840637207]], [[4.2484869956970215]], [[4.344150066375732]], [[4.911170482635498]], [[4.827535629272461]], [[4.324012279510498]], [[4.877218723297119]], [[4.857936382293701]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_47d13bb7e11c7a43969e46d30fed53ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.710595607757568]], [[5.883352756500244]], [[5.962912082672119]], [[6.112746715545654]], [[6.351295471191406]], [[5.6259589195251465]], [[5.555627346038818]], [[6.105219841003418]], [[5.789981842041016]], [[6.86287784576416]], [[5.55156135559082]], [[5.655170440673828]], [[5.961136341094971]], [[5.718842506408691]], [[5.831554412841797]], [[5.7848310470581055]], [[6.288494110107422]], [[5.579275608062744]], [[6.376399517059326]], [[6.480332374572754]], [[5.778118133544922]], [[5.190840244293213]], [[6.211253643035889]], [[5.805886745452881]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_14cd734d2ae2ea6014bece77e47919ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.17759385704994202]], [[0.19153180718421936]], [[0.3440287411212921]], [[0.008298016153275967]], [[0.17216309905052185]], [[0.20801407098770142]], [[0.11266348510980606]], [[0.04081058129668236]], [[0.09205195307731628]], [[0.49273884296417236]], [[0.05018388852477074]], [[0.0915977954864502]], [[0.1720375418663025]], [[0.32620975375175476]], [[0.008654437027871609]], [[0.10290484875440598]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.uniform([4, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2aca9f646495dec1fb8b3bc83cffebe1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9315390586853027]], [[1.179887294769287]], [[1.1326204538345337]], [[1.0238231420516968]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


class TestPrimitiveOp_6841f983fe1f5500bd124d53688cab33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.075436592102051]], [[3.1854355335235596]], [[2.4594452381134033]], [[3.3009822368621826]], [[2.8738415241241455]], [[3.0226480960845947]], [[3.174708127975464]], [[3.2025771141052246]], [[2.724499225616455]], [[3.0178182125091553]], [[2.595012903213501]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


class TestPrimitiveOp_6d85d72349d4f02b0a53c983bbf339f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.356210708618164]], [[7.45643949508667]], [[7.715261936187744]], [[7.937924385070801]], [[7.409104824066162]], [[8.703195571899414]], [[6.576315879821777]], [[8.27151870727539]], [[7.770561695098877]], [[7.652752876281738]], [[7.357721328735352]], [[7.527255058288574]], [[7.71347713470459]], [[6.793307304382324]], [[7.76899528503418]], [[7.068814277648926]], [[8.558565139770508]], [[7.061863422393799]], [[7.1605939865112305]], [[6.982356071472168]], [[7.739774703979492]], [[8.013056755065918]], [[7.885787487030029]], [[7.428193092346191]], [[6.9627861976623535]], [[8.001039505004883]], [[8.861226081848145]], [[7.721975326538086]], [[7.300220012664795]], [[7.781312465667725]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_e3daf7362893253019db92119e8decd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.852346420288086]], [[3.617408275604248]], [[3.7377684116363525]], [[4.023648738861084]], [[3.9766042232513428]], [[3.867718458175659]], [[3.8445804119110107]], [[4.549490451812744]], [[4.194453239440918]], [[3.9696319103240967]], [[3.9156904220581055]], [[3.870849132537842]], [[4.273193359375]], [[4.206573009490967]], [[4.431917190551758]], [[3.9569289684295654]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_2486f1e8027ead37f1a3125ce01c0224(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 7581, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.3343918025493622]], [[0.08188877999782562]], [[0.22011376917362213]], [[0.2466224581003189]], [[0.3462381958961487]], [[0.0736234188079834]], [[0.2025250792503357]], [[0.12420257180929184]], [[0.43870195746421814]], [[0.11114651709794998]], [[0.03658818081021309]], [[0.3381662666797638]], [[0.48259204626083374]], [[0.25881102681159973]], [[0.2802067995071411]], [[0.16233551502227783]], [[0.1209297925233841]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_dfd769d694696996b7af6f68fe51ee61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 4725, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.45448845624923706]], [[0.21619829535484314]], [[0.04356005787849426]], [[0.37630629539489746]], [[0.10138637572526932]], [[0.02058848738670349]], [[0.0935359001159668]], [[0.22291956841945648]], [[0.001122866291552782]], [[0.18888655304908752]], [[0.27897313237190247]], [[0.45071348547935486]], [[0.2856830656528473]], [[0.11291929334402084]], [[0.31318020820617676]], [[0.23970453441143036]], [[0.20238231122493744]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_42323315d7b4007f71f13c3501fa3e79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.047674179077148]], [[7.46746301651001]], [[7.71571159362793]], [[7.386870861053467]], [[6.661334991455078]], [[7.3748087882995605]], [[7.907918930053711]], [[7.6311774253845215]], [[7.681912422180176]], [[8.161060333251953]], [[7.657646656036377]], [[7.225220203399658]], [[8.0458984375]], [[7.048724174499512]], [[7.61449670791626]], [[8.095491409301758]], [[7.270317077636719]], [[7.645841121673584]], [[7.96390438079834]], [[7.035454750061035]], [[7.8601603507995605]], [[8.187243461608887]], [[7.682562351226807]], [[6.853982448577881]], [[7.161929130554199]], [[7.546895503997803]], [[6.980509281158447]], [[7.240118980407715]], [[8.376116752624512]], [[7.675683498382568]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_ff2b88d2b5cfe90de8a5bcd3301007c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.161099433898926]], [[6.283670425415039]], [[7.077556610107422]], [[6.608377933502197]], [[6.8887434005737305]], [[7.144110202789307]], [[7.002241611480713]], [[6.787414073944092]], [[6.6316022872924805]], [[7.379030704498291]], [[7.145711421966553]], [[7.1580424308776855]], [[5.669384002685547]], [[6.9816694259643555]], [[7.313919544219971]], [[7.1442108154296875]], [[7.372506141662598]], [[6.458278179168701]], [[6.062782287597656]], [[6.468989849090576]], [[7.340620517730713]], [[6.763891696929932]], [[7.095375061035156]], [[6.321043014526367]], [[6.622008323669434]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_002ac61c978e124b85cb216b630558d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 8400, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2660098075866699]], [[0.10283896327018738]], [[0.3085457980632782]], [[0.4746760427951813]], [[0.26985296607017517]], [[0.10924237221479416]], [[0.06053238362073898]], [[0.23296622931957245]], [[0.44863399863243103]], [[0.23619070649147034]], [[0.1294732689857483]], [[0.015674348920583725]], [[0.38847029209136963]], [[0.14438322186470032]], [[0.08365380018949509]], [[0.37206825613975525]], [[0.1498025506734848]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_b05a1489f5af94ce4d6a00923350b58b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.157277584075928]], [[6.654095649719238]], [[6.1927008628845215]], [[6.174805164337158]], [[5.656250953674316]], [[6.049745559692383]], [[7.229335308074951]], [[5.894186973571777]], [[5.981937885284424]], [[6.447387218475342]], [[5.8650946617126465]], [[5.305336952209473]], [[6.0240936279296875]], [[6.597114562988281]], [[6.157079696655273]], [[6.523933410644531]], [[5.694827079772949]], [[6.395604133605957]], [[6.7870025634765625]], [[5.419206142425537]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_fe1c095fcb0129673a05171226d59ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.003097210545092821]], [[0.15708604454994202]], [[0.036929477006196976]], [[0.4361492097377777]], [[0.19270308315753937]], [[0.13369183242321014]], [[0.07259334623813629]], [[0.4134281873703003]], [[0.20328085124492645]], [[0.4855978786945343]], [[0.10756264626979828]], [[0.2658918499946594]], [[0.38615015149116516]], [[0.2525254189968109]], [[0.18120674788951874]], [[0.34687715768814087]], [[0.1878626048564911]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_5996079ac80d5d8cbde18258ce91df3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.10752534866333]], [[4.603259563446045]], [[4.440310478210449]], [[4.605095863342285]], [[4.184913158416748]], [[3.8676671981811523]], [[5.142180919647217]], [[5.176795482635498]], [[4.383977890014648]], [[3.788083076477051]], [[4.922099590301514]], [[4.851803779602051]], [[4.568479537963867]], [[4.971857070922852]], [[4.623622417449951]], [[4.299269199371338]], [[4.922940254211426]], [[3.8281266689300537]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_82565c72c1ab56daf0169c7d04871a37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.549908638000488]], [[4.9383721351623535]], [[4.530771732330322]], [[4.003131866455078]], [[4.2931389808654785]], [[4.7434468269348145]], [[4.119290351867676]], [[4.080099582672119]], [[4.670665740966797]], [[4.28275728225708]], [[4.067380428314209]], [[4.824221134185791]], [[4.491727352142334]], [[5.19081974029541]], [[4.20680570602417]], [[4.18110466003418]], [[4.830199241638184]], [[4.654162883758545]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_6c1874235a45757124ffb3b5c9a1e882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.730341911315918]], [[5.2150139808654785]], [[5.373372554779053]], [[5.788073539733887]], [[6.086791515350342]], [[6.717208385467529]], [[5.628145694732666]], [[5.626158237457275]], [[5.946242809295654]], [[5.797774791717529]], [[5.717459678649902]], [[5.4825263023376465]], [[5.418801307678223]], [[5.495123863220215]], [[5.399899959564209]], [[6.1512064933776855]], [[6.103494644165039]], [[5.643342018127441]], [[5.740949630737305]], [[5.626522064208984]], [[6.405750751495361]], [[6.131895542144775]], [[5.576231002807617]], [[6.0827131271362305]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_f4143b8db70acf1f1e747bcda9511486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.239541053771973]], [[4.816692352294922]], [[4.858314514160156]], [[5.074556350708008]], [[4.58089542388916]], [[4.850882053375244]], [[4.191903591156006]], [[4.253087520599365]], [[4.150959014892578]], [[3.8840956687927246]], [[4.621192932128906]], [[4.307068347930908]], [[5.264896392822266]], [[4.96284818649292]], [[4.513309001922607]], [[4.3990349769592285]], [[4.5447516441345215]], [[4.599179744720459]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_e9f9b6c079c004233a23143e6891ba20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.579036712646484]], [[5.627302646636963]], [[5.249024391174316]], [[6.085101127624512]], [[5.456614971160889]], [[5.158232688903809]], [[5.729126453399658]], [[5.048123359680176]], [[5.283633232116699]], [[4.933670520782471]], [[5.621869087219238]], [[4.86514139175415]], [[5.098511695861816]], [[5.634434223175049]], [[5.756320953369141]], [[5.218016147613525]], [[4.7758941650390625]], [[5.290668964385986]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_ca99d5c8eaf0b24aadf83d6a239cb30a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 4116, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1629117876291275]], [[0.4102975130081177]], [[0.43351539969444275]], [[0.31436029076576233]], [[0.17669720947742462]], [[0.2512248158454895]], [[0.2866753935813904]], [[0.20648223161697388]], [[0.25126582384109497]], [[0.34065672755241394]], [[0.18669411540031433]], [[0.13551567494869232]], [[0.06203129515051842]], [[0.2872585952281952]], [[0.47710713744163513]], [[0.06563885509967804]], [[0.029908401891589165]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_a98b8c9d30366c9bf4b7df61bef731ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 6069, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.365415096282959]], [[0.21937866508960724]], [[0.4881259799003601]], [[0.0022065152879804373]], [[0.10819203406572342]], [[0.19719719886779785]], [[0.01275209616869688]], [[0.07600267231464386]], [[0.28773245215415955]], [[0.31463193893432617]], [[0.2582530975341797]], [[0.3900698721408844]], [[0.20377472043037415]], [[0.03238197788596153]], [[0.0771549642086029]], [[0.3871915936470032]], [[0.21753545105457306]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_333c4727fd3e4a58adb12150b9a8f0ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.774139881134033]], [[4.44288444519043]], [[4.061627388000488]], [[4.734555721282959]], [[4.272776126861572]], [[4.279536724090576]], [[4.924454689025879]], [[4.579849720001221]], [[5.035876274108887]], [[4.642727375030518]], [[5.036672592163086]], [[4.533378601074219]], [[4.076155662536621]], [[4.543557643890381]], [[4.444705486297607]], [[5.03630256652832]], [[4.620931148529053]], [[4.428459167480469]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_b062fc6f456b9f536826250c2d7a75ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.887784004211426]], [[3.400664806365967]], [[3.332968235015869]], [[4.292311668395996]], [[3.8454418182373047]], [[3.7534611225128174]], [[3.7757980823516846]], [[3.8074114322662354]], [[3.3374392986297607]], [[3.346233606338501]], [[4.2629899978637695]], [[3.754770517349243]], [[3.4879889488220215]], [[3.4377477169036865]], [[2.8427469730377197]], [[3.731217861175537]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_56682734c0ec01fca69c879dee014239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.966066360473633]], [[4.719254016876221]], [[5.574428558349609]], [[5.309321880340576]], [[5.40964412689209]], [[5.289345741271973]], [[5.064115047454834]], [[4.791955947875977]], [[4.5315470695495605]], [[4.723508834838867]], [[4.917330265045166]], [[5.320054531097412]], [[4.874161720275879]], [[5.4857330322265625]], [[5.171445369720459]], [[5.232664585113525]], [[4.794351577758789]], [[5.000119209289551]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_f5fb23d87192d2b45ff96177db0d0e3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.26943114399909973]], [[0.3764727711677551]], [[0.042164869606494904]], [[0.2340172976255417]], [[0.3209754228591919]], [[0.001512373797595501]], [[0.2309127002954483]], [[0.2170521318912506]], [[0.22433793544769287]], [[0.48192858695983887]], [[0.4640029966831207]], [[0.08406952023506165]], [[0.2777986526489258]], [[0.38330328464508057]], [[0.2529655694961548]], [[0.1469050496816635]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.uniform([4, 16, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d36fb79d4340f73f73ba972c08dabce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3242740631103516]], [[1.1619985103607178]], [[1.2257239818572998]], [[1.600968837738037]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


class TestPrimitiveOp_c7506b4c5081627d4bc7fa45b4aea372(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.867608547210693]], [[5.49536657333374]], [[5.5151143074035645]], [[5.808355808258057]], [[5.518977165222168]], [[5.096158504486084]], [[6.271181106567383]], [[5.590792655944824]], [[5.520811080932617]], [[5.452767848968506]], [[5.823544502258301]], [[5.6400346755981445]], [[5.126609802246094]], [[5.012604236602783]], [[5.147573471069336]], [[5.880778789520264]], [[4.466744899749756]], [[5.142038822174072]], [[5.521035194396973]], [[5.765078067779541]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_78962302d0c9ef463ebad22f30d848bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.670431613922119]], [[3.4017767906188965]], [[3.7088985443115234]], [[2.8188748359680176]], [[3.511875629425049]], [[3.398378849029541]], [[3.6883749961853027]], [[3.3284430503845215]], [[3.064164876937866]], [[3.123818874359131]], [[3.139251232147217]], [[3.20597243309021]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


class TestPrimitiveOp_d008d50b6392aef7e6d4b802f53fbed5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.653189659118652]], [[4.214825630187988]], [[5.99698543548584]], [[5.331616401672363]], [[5.504478931427002]], [[5.55187463760376]], [[5.197627544403076]], [[5.065846920013428]], [[5.104503631591797]], [[5.2176194190979]], [[5.302167892456055]], [[4.6718244552612305]], [[4.829518795013428]], [[5.492980003356934]], [[4.358083248138428]], [[5.401058197021484]], [[5.749157905578613]], [[4.652544975280762]], [[5.41537618637085]], [[5.704509258270264]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_d028be892621ee825dd122e7cdffe498(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.541774272918701]], [[3.583669900894165]], [[3.2327818870544434]], [[3.3782896995544434]], [[2.9439098834991455]], [[3.5128774642944336]], [[3.4546477794647217]], [[3.4707159996032715]], [[2.7985730171203613]], [[3.3343186378479004]], [[2.6518492698669434]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


class TestPrimitiveOp_eb6cfca72b928fe4298338bbee229e8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.9662227630615234]], [[3.4809980392456055]], [[3.050659418106079]], [[3.953287124633789]], [[3.764752149581909]], [[3.1874589920043945]], [[3.9026811122894287]], [[4.052516937255859]], [[3.5598158836364746]], [[3.4012203216552734]], [[2.9586782455444336]], [[3.6440930366516113]], [[3.2894439697265625]], [[3.7816214561462402]]]], dtype='float32').reshape([1, 14, 1, 1]),
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


class TestPrimitiveOp_d993433ec77829ce2b659b55c0d47cd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.126227378845215]], [[4.615624904632568]], [[5.39639949798584]], [[5.142935276031494]], [[5.045383453369141]], [[4.894627571105957]], [[5.579664707183838]], [[4.875969409942627]], [[5.422024726867676]], [[4.741666316986084]], [[5.0650954246521]], [[5.370575428009033]], [[4.293694496154785]], [[4.9663920402526855]], [[5.268765449523926]], [[4.810600280761719]], [[4.8917646408081055]], [[4.197942733764648]], [[5.488993167877197]], [[4.756964683532715]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_3d43b5fb2b0bd4ba59cdb0de58b160a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[42277.2890625]], [[34658.6796875]], [[37588.92578125]], [[32222.041015625]], [[29644.5390625]], [[33550.40625]]], [[[41074.41796875]], [[33686.07421875]], [[36531.6640625]], [[31303.294921875]], [[28817.443359375]], [[32597.2421875]]]], dtype='float32').reshape([2, 6, 1, 1]),
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


class TestPrimitiveOp_7ebb7c99d87e85621a7aea1fc80e626b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[37649.17578125]], [[41821.875]], [[33838.6640625]], [[36712.5859375]], [[36409.48046875]], [[35812.41796875]]], [[[38640.72265625]], [[42923.26953125]], [[34727.796875]], [[37678.38671875]], [[37369.30078125]], [[36757.06640625]]]], dtype='float32').reshape([2, 6, 1, 1]),
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


class TestPrimitiveOp_12e5fc236edb1e42e73e1782a539b7f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[45982.49609375]], [[30480.904296875]], [[38091.53125]], [[41839.31640625]], [[38919.12890625]], [[43576.6953125]]], [[[46880.86328125]], [[31078.677734375]], [[38840.7890625]], [[42663.890625]], [[39683.58984375]], [[44432.76953125]]]], dtype='float32').reshape([2, 6, 1, 1]),
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


class TestPrimitiveOp_7c55e256ba89130fc6ae476f3d72e358(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[45499.42578125]], [[44213.3515625]], [[37121.56640625]], [[43967.328125]], [[41490.19921875]], [[45803.73828125]]], [[[45880.5234375]], [[44590.3203125]], [[37436.80078125]], [[44341.53125]], [[41841.59765625]], [[46192.51171875]]]], dtype='float32').reshape([2, 6, 1, 1]),
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


class TestPrimitiveOp_9af915ae9d4950621a16d4ebf5bbc46d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 9261, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.004111988935619593]], [[0.2624770700931549]], [[0.027654627338051796]], [[0.30951082706451416]], [[0.24769191443920135]], [[0.43815523386001587]], [[0.2938857078552246]], [[0.35976842045783997]], [[0.3157884180545807]], [[0.3585507273674011]], [[0.46862003207206726]], [[0.2218005359172821]], [[0.2945643365383148]], [[0.48500680923461914]], [[0.49645665287971497]], [[0.005673353094607592]], [[0.1333989053964615]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_26f3619eb584231f10e493e2f1de1f7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.542322635650635]], [[7.213070869445801]], [[6.877016067504883]], [[7.558468818664551]], [[7.462998867034912]], [[7.755030632019043]], [[7.1506733894348145]], [[7.84716272354126]], [[7.4940876960754395]], [[7.207822322845459]], [[7.077805042266846]], [[7.42388916015625]], [[7.746916770935059]], [[7.274901866912842]], [[6.521085739135742]], [[7.162106990814209]], [[6.552116870880127]], [[6.899572849273682]], [[7.4951395988464355]], [[7.841178894042969]], [[8.506115913391113]], [[7.153757095336914]], [[7.740911483764648]], [[7.616919040679932]], [[7.1264472007751465]], [[7.494256973266602]], [[7.5830464363098145]], [[7.284029483795166]], [[8.21269702911377]], [[7.506575107574463]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_542b96b9f6c0c2d9a0f715667c7b0b02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[9.350680351257324]], [[8.810539245605469]], [[8.99864387512207]], [[8.117464065551758]], [[8.394437789916992]], [[8.087057113647461]], [[8.953898429870605]], [[8.628273010253906]], [[8.878339767456055]], [[8.572881698608398]], [[8.787232398986816]], [[7.157745838165283]], [[8.177393913269043]], [[8.811300277709961]], [[9.26130485534668]], [[7.19333028793335]], [[9.088506698608398]], [[9.55791187286377]], [[8.022857666015625]], [[8.532830238342285]], [[9.058215141296387]], [[7.555534362792969]], [[8.369180679321289]], [[7.965431213378906]], [[9.163026809692383]], [[10.197159767150879]], [[9.149077415466309]], [[7.875706672668457]], [[8.75112533569336]], [[8.162129402160645]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_c32102b08eabc80de25791f77d269eeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.295289516448975]], [[7.204415321350098]], [[7.707129001617432]], [[7.5969390869140625]], [[7.645872592926025]], [[7.568965435028076]], [[8.013806343078613]], [[7.534467697143555]], [[7.524115562438965]], [[7.817281246185303]], [[7.21710205078125]], [[6.7774739265441895]], [[7.938440799713135]], [[7.088042736053467]], [[7.350950717926025]], [[7.004399299621582]], [[7.458910942077637]], [[6.457792282104492]], [[6.906866550445557]], [[7.151056289672852]], [[7.212295055389404]], [[7.2559285163879395]], [[7.21386194229126]], [[7.163641452789307]], [[6.932377815246582]], [[7.226587295532227]], [[7.675450801849365]], [[7.0391845703125]], [[7.504758358001709]], [[6.9274210929870605]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_a278266068f495a7d5172d0600d3bd03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.369719505310059]], [[7.840510845184326]], [[8.52347469329834]], [[7.7876739501953125]], [[8.8694486618042]], [[7.973025798797607]], [[8.4177885055542]], [[9.125961303710938]], [[7.797096252441406]], [[8.53489875793457]], [[7.907627105712891]], [[8.323229789733887]], [[8.71175765991211]], [[8.683496475219727]], [[8.667680740356445]], [[8.16464900970459]], [[8.238906860351562]], [[7.278522491455078]], [[9.046670913696289]], [[7.9590654373168945]], [[8.00121021270752]], [[7.9140191078186035]], [[7.0355377197265625]], [[8.831998825073242]], [[7.775221824645996]], [[8.064579010009766]], [[8.203347206115723]], [[7.545536518096924]], [[7.598377704620361]], [[9.136058807373047]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_091ee118371a5b28a3991576523b54b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.9090192317962646]], [[3.3313701152801514]], [[3.1470601558685303]], [[3.0634727478027344]], [[3.3448996543884277]], [[2.3325366973876953]], [[2.646023750305176]], [[2.83821964263916]], [[3.1006522178649902]], [[3.114042282104492]], [[2.988996744155884]], [[2.8920671939849854]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


class TestPrimitiveOp_73b281664e564d9ccde2bc4fbefa33b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.6413681507110596]], [[3.0041394233703613]], [[2.860701322555542]], [[2.8276419639587402]], [[3.354630947113037]], [[2.4001946449279785]], [[2.765003204345703]], [[2.947614908218384]], [[2.6335387229919434]], [[3.0719587802886963]], [[3.2401950359344482]], [[2.63741135597229]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


class TestPrimitiveOp_3215fea076fbd3c333b3671e84459392(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.43465518951416]], [[6.167778491973877]], [[6.1945672035217285]], [[5.861127853393555]], [[6.375700950622559]], [[6.514279365539551]], [[5.694643974304199]], [[6.402563095092773]], [[6.252643585205078]], [[5.755939960479736]], [[7.040493965148926]], [[6.350912094116211]], [[6.256567478179932]], [[7.086465835571289]], [[6.296385765075684]], [[7.195964336395264]], [[5.541567325592041]], [[6.9893059730529785]], [[5.531717300415039]], [[5.982404708862305]], [[6.592355728149414]], [[5.684733867645264]], [[5.915470600128174]], [[5.6908860206604]], [[6.603821754455566]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_cd330e3ab0b2815eb28866a0beb1472c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 2100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.22066178917884827]], [[0.03212594613432884]], [[0.11090849339962006]], [[0.2760164439678192]], [[0.15171702206134796]], [[0.1631331443786621]], [[0.2525159418582916]], [[0.22890159487724304]], [[0.28260111808776855]], [[0.4575783908367157]], [[0.34583789110183716]], [[0.3152821958065033]], [[0.14452864229679108]], [[0.206071138381958]], [[0.4516810178756714]], [[0.42708417773246765]], [[0.017734384164214134]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_6beb2a302b39f9adbc13d7aa597770c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.360105991363525]], [[4.688514709472656]], [[4.720791816711426]], [[4.639763355255127]], [[4.480045795440674]], [[4.889379501342773]], [[4.068061351776123]], [[4.908852577209473]], [[4.4659833908081055]], [[5.004398822784424]], [[5.070291042327881]], [[5.3074212074279785]], [[4.330776691436768]], [[4.965434551239014]], [[3.9914205074310303]], [[4.536936283111572]], [[4.425126552581787]], [[4.817387104034424]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_d29b4a95ec8486e21f88ade3a5bc2280(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.05157182365655899]], [[0.37332189083099365]], [[0.10245512425899506]], [[0.380955308675766]], [[0.2025565356016159]], [[0.365032434463501]], [[0.4064849019050598]], [[0.0435660295188427]], [[0.09678394347429276]], [[0.21878477931022644]], [[0.26359662413597107]], [[0.24752925336360931]], [[0.46972060203552246]], [[0.2761533558368683]], [[0.4291585087776184]], [[0.1368430256843567]], [[0.0053528184071183205]], [[0.010435297153890133]], [[0.09848395735025406]], [[0.26137158274650574]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([5, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe69f31e1cdcee92a7bae4a74165ae46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.690147876739502]], [[1.1477481126785278]], [[1.5876469612121582]], [[1.4030735492706299]], [[1.603986382484436]]]], dtype='float32').reshape([1, 5, 1, 1]),
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


class TestPrimitiveOp_79736bbef253e6cb22178f060f03ee14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.1300649642944336]], [[3.0645670890808105]], [[2.788966178894043]], [[2.701542615890503]], [[2.479606866836548]], [[2.1792891025543213]], [[2.565462112426758]], [[2.7054295539855957]], [[2.5611631870269775]], [[2.8416919708251953]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_96fe050f5a03416654ee2ac260a102f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.086889743804932]], [[5.554097652435303]], [[5.618411540985107]], [[5.302731037139893]], [[5.082913875579834]], [[5.409112930297852]], [[4.88266658782959]], [[5.384340286254883]], [[5.555750370025635]], [[4.858942031860352]], [[5.871918201446533]], [[5.652488708496094]], [[4.972433567047119]], [[5.516233444213867]], [[5.493321895599365]], [[4.793114185333252]], [[6.127239227294922]], [[5.900052070617676]], [[5.483379364013672]], [[5.497974395751953]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_02cc35811979c74d58a9de7577f50bd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 11109, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.011151811107993126]], [[0.30401018261909485]], [[0.06690847128629684]], [[0.45153898000717163]], [[0.3324783146381378]], [[0.17745056748390198]], [[0.4116470515727997]], [[0.3507973849773407]], [[0.016656674444675446]], [[0.06560871005058289]], [[0.4102177023887634]], [[0.4879305362701416]], [[0.4060313403606415]], [[0.14999037981033325]], [[0.3466957211494446]], [[0.2642812430858612]], [[0.056544870138168335]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_13faa74c57887c519e893a6272f4c05a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.806848049163818]], [[6.779421806335449]], [[5.928939342498779]], [[6.151634216308594]], [[5.8947577476501465]], [[5.825343132019043]], [[6.187404155731201]], [[5.690608501434326]], [[6.611276626586914]], [[7.166670799255371]], [[7.097313404083252]], [[6.954497337341309]], [[6.246237754821777]], [[6.205399036407471]], [[5.889923572540283]], [[6.354040622711182]], [[6.265686988830566]], [[6.760129928588867]], [[6.005337238311768]], [[6.041594505310059]], [[6.07476806640625]], [[5.392578125]], [[6.428365707397461]], [[6.381635665893555]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_5ab3d4d5cd86e417982df67bfafc95be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.7149548530578613]], [[2.836534023284912]], [[2.2737889289855957]], [[2.4234912395477295]], [[2.16805362701416]], [[2.763005018234253]], [[3.5774857997894287]], [[1.8416321277618408]], [[2.503131866455078]], [[2.6240639686584473]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_7f166866ae08f04c7b61e03952e01a7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.989027976989746]], [[4.333471298217773]], [[4.135684013366699]], [[5.009162902832031]], [[4.20782470703125]], [[4.118492126464844]], [[4.63630485534668]], [[4.8843817710876465]], [[4.670323848724365]], [[4.553167343139648]], [[4.147002220153809]], [[4.106956958770752]], [[4.478455066680908]], [[4.616498947143555]], [[4.093318939208984]], [[5.0294952392578125]], [[4.685277462005615]], [[5.693291187286377]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


class TestPrimitiveOp_ff31146d0289d5190c372632d51a3f50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.190667152404785]], [[8.532549858093262]], [[7.578472137451172]], [[8.632094383239746]], [[7.576027870178223]], [[8.524397850036621]], [[7.91270637512207]], [[7.142776012420654]], [[8.293561935424805]], [[6.992138862609863]], [[7.605541229248047]], [[7.522144794464111]], [[7.927464008331299]], [[7.343049049377441]], [[8.28248119354248]], [[7.635629653930664]], [[7.863544940948486]], [[8.674365997314453]], [[8.02641773223877]], [[6.868565559387207]], [[7.765742778778076]], [[7.794824600219727]], [[7.496888637542725]], [[7.5954389572143555]], [[7.824265956878662]], [[7.748753070831299]], [[7.126996994018555]], [[7.634503364562988]], [[6.999751091003418]], [[8.412819862365723]]]], dtype='float32').reshape([1, 30, 1, 1]),
            paddle.uniform([120, 30, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7bd5687b653c461e4ecaad40fe0426e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.45764294266700745]], [[0.3116616904735565]], [[0.23043279349803925]], [[0.2746705412864685]], [[0.13954804837703705]], [[0.08488123118877411]], [[0.0587196871638298]], [[0.11238455027341843]], [[0.25665247440338135]], [[0.16044490039348602]], [[0.18880541622638702]], [[0.3744608759880066]], [[0.3700231611728668]], [[0.24264194071292877]], [[0.11436773836612701]], [[0.23741501569747925]], [[0.41314762830734253]], [[0.18248794972896576]], [[0.4323033094406128]], [[0.11006316542625427]]]], dtype='float32').reshape([1, 20, 1, 1]),
            paddle.uniform([5, 20, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df1547a370add3a5288520d59231b8a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.117078423500061]], [[1.4652667045593262]], [[1.2384942770004272]], [[1.3474791049957275]], [[1.5838005542755127]]]], dtype='float32').reshape([1, 5, 1, 1]),
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


class TestPrimitiveOp_0e92214ca771e2dacc0dd3f51f7a35f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.344630718231201]], [[2.803612232208252]], [[2.660125732421875]], [[2.699988842010498]], [[2.7060811519622803]], [[2.6608357429504395]], [[2.113220453262329]], [[2.41262149810791]], [[2.4208219051361084]], [[2.095384359359741]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


class TestPrimitiveOp_a12bb20bddbada664d7131689d280532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.292038440704346]], [[5.7622833251953125]], [[5.8628973960876465]], [[5.231494426727295]], [[5.926563739776611]], [[5.8816022872924805]], [[4.67780065536499]], [[4.886355876922607]], [[5.886889457702637]], [[5.720672130584717]], [[5.529699802398682]], [[6.2631025314331055]], [[6.021288871765137]], [[5.579195022583008]], [[5.917940616607666]], [[4.742470741271973]], [[5.772171497344971]], [[4.694761276245117]], [[5.614266872406006]], [[5.412294864654541]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_09eb1fd106cc4e9d645b2ec234d7a794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.885272979736328]], [[3.995682716369629]], [[4.380467891693115]], [[4.238519668579102]], [[3.807973861694336]], [[3.770780086517334]], [[3.4668946266174316]], [[3.885648250579834]], [[3.979804039001465]], [[3.3920586109161377]], [[4.023132801055908]], [[3.702303409576416]], [[3.5543558597564697]], [[3.9580507278442383]], [[4.54077672958374]], [[4.028207302093506]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_f4f4f8171bd480fa9a73c24aba4dd51f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.51597261428833]], [[3.731383800506592]], [[3.5915372371673584]], [[3.3545918464660645]], [[3.65403151512146]], [[3.4771029949188232]], [[3.5580673217773438]], [[3.1740853786468506]], [[3.2493107318878174]], [[3.4338717460632324]], [[3.2165613174438477]], [[3.9413301944732666]], [[3.2475860118865967]], [[3.414360523223877]]]], dtype='float32').reshape([1, 14, 1, 1]),
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


class TestPrimitiveOp_09f7cc36643464f1f28fc19fc1857b77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.418937683105469]], [[5.241125106811523]], [[5.984535217285156]], [[5.723231792449951]], [[5.40896463394165]], [[5.961812973022461]], [[5.400176048278809]], [[5.480480670928955]], [[6.17966365814209]], [[6.024230003356934]], [[5.300405502319336]], [[6.840229511260986]], [[5.395784378051758]], [[5.9628729820251465]], [[5.627788543701172]], [[5.5482025146484375]], [[5.286175727844238]], [[5.988452911376953]], [[5.808948516845703]], [[5.572681427001953]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_4994d4390ff2830ae81a05c97707f119(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.764605522155762]], [[7.043685436248779]], [[7.240815162658691]], [[7.279595851898193]], [[6.5231523513793945]], [[7.445433616638184]], [[6.549773693084717]], [[7.3310017585754395]], [[7.319559097290039]], [[7.149440765380859]], [[6.387151718139648]], [[7.872252464294434]], [[7.2891998291015625]], [[8.173873901367188]], [[7.665742874145508]], [[6.962970733642578]], [[6.7176642417907715]], [[7.236079216003418]], [[8.199427604675293]], [[7.267073154449463]], [[7.563003063201904]], [[7.959856033325195]], [[6.962826251983643]], [[7.467273235321045]], [[6.880935192108154]], [[6.395427703857422]], [[7.35816764831543]], [[7.327852249145508]], [[7.032649517059326]], [[7.583899974822998]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


class TestPrimitiveOp_8c72b77b8d4cc094eb221b1d05661b33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 3024, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.350218266248703]], [[0.435467392206192]], [[0.05841661989688873]], [[0.28512808680534363]], [[0.41358184814453125]], [[0.4594303369522095]], [[0.36511844396591187]], [[0.1673954725265503]], [[0.17076238989830017]], [[0.04395011439919472]], [[0.41647136211395264]], [[0.014523189514875412]], [[0.04294595122337341]], [[0.4961397647857666]], [[0.14285363256931305]], [[0.46566247940063477]], [[0.3618255853652954]]]], dtype='float32').reshape([1, 17, 1, 1]),
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


class TestPrimitiveOp_bbd63581b02aa08a9cd7176c843dcdba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.254993915557861]], [[6.970305442810059]], [[6.205292701721191]], [[6.2595343589782715]], [[5.769108772277832]], [[5.224167346954346]], [[6.379481792449951]], [[5.747693061828613]], [[5.659907817840576]], [[5.943023204803467]], [[6.025867462158203]], [[5.703178882598877]], [[5.720294952392578]], [[6.185833930969238]], [[5.226645469665527]], [[6.315010070800781]], [[6.4829301834106445]], [[6.07181978225708]], [[5.83260440826416]], [[6.747703552246094]], [[5.934568405151367]], [[6.243119239807129]], [[6.224294185638428]], [[5.637502670288086]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_980fae2135ee49703206cc970f3d72ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.551765441894531]], [[6.629490852355957]], [[5.867996692657471]], [[6.1684465408325195]], [[6.284104347229004]], [[6.519019603729248]], [[6.52275276184082]], [[6.112610816955566]], [[6.900456428527832]], [[6.185481071472168]], [[6.595438003540039]], [[6.794672012329102]], [[6.027585029602051]], [[6.001448631286621]], [[6.552099704742432]], [[5.967398643493652]], [[6.044076442718506]], [[6.4735331535339355]], [[6.647521495819092]], [[6.414856910705566]], [[6.089977741241455]], [[6.816417694091797]], [[5.752189636230469]], [[6.891372203826904]], [[6.464967250823975]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


class TestPrimitiveOp_f49a2b0747c5976ba1b8ca728d495e03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.279315710067749]], [[3.2554545402526855]], [[3.3334555625915527]], [[3.2763848304748535]], [[3.1578352451324463]], [[3.2209408283233643]], [[3.4248428344726562]], [[3.2314765453338623]], [[3.383251667022705]], [[3.0282557010650635]], [[3.131232738494873]], [[3.0485799312591553]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


class TestPrimitiveOp_855b207498595b9fdc81ad178a7cc805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[738.3297729492188]], [[772.1773071289062]], [[697.5306396484375]], [[757.9332275390625]], [[706.9006958007812]], [[710.8587036132812]], [[772.3379516601562]], [[776.5546264648438]], [[747.6640625]], [[698.0125732421875]], [[702.9410400390625]], [[719.0142211914062]], [[732.46142578125]], [[575.9503173828125]], [[770.2093505859375]], [[795.2775268554688]], [[707.408203125]], [[702.1323852539062]], [[775.8509521484375]], [[672.6448974609375]], [[686.5088500976562]], [[694.6845092773438]], [[732.1170043945312]], [[839.0545043945312]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_fec313f5f95693010d55ee25f13852f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[88.27577209472656]], [[95.76002502441406]], [[87.78861999511719]], [[87.79698944091797]], [[88.5737533569336]], [[80.44439697265625]], [[89.31011962890625]], [[87.85270690917969]], [[84.488525390625]], [[89.33270263671875]], [[88.08332061767578]], [[84.07212829589844]], [[97.84424591064453]], [[85.0938720703125]], [[89.42194366455078]], [[83.89889526367188]], [[91.36042022705078]], [[87.6751937866211]], [[82.2608413696289]], [[94.42173767089844]], [[97.65007019042969]], [[84.28276824951172]], [[90.93608856201172]], [[96.48577117919922]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_5940ef51690d1bbbda04d3c72ac4d580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[36.361244201660156]], [[35.59650421142578]], [[35.80752944946289]], [[36.66264343261719]], [[36.12427520751953]], [[30.409032821655273]], [[35.09974670410156]], [[34.62222671508789]], [[32.15555191040039]], [[31.91900062561035]], [[31.293312072753906]], [[36.2292366027832]], [[36.25039291381836]], [[37.37382125854492]], [[36.91184616088867]], [[35.765750885009766]], [[34.373870849609375]], [[37.94071578979492]], [[36.336631774902344]], [[34.920448303222656]], [[34.973148345947266]], [[35.75593948364258]], [[35.398006439208984]], [[35.97702407836914]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_dcef0a4f51762c4065ed4a29a5081612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[25.351303100585938]], [[27.760879516601562]], [[31.212581634521484]], [[29.61737632751465]], [[27.91501808166504]], [[28.912965774536133]], [[28.17799949645996]], [[27.986726760864258]], [[29.23400115966797]], [[25.430774688720703]], [[28.742856979370117]], [[26.738697052001953]], [[24.9879207611084]], [[23.688814163208008]], [[28.042436599731445]], [[28.10509490966797]], [[27.81442642211914]], [[31.373493194580078]], [[29.423311233520508]], [[26.931270599365234]], [[27.819124221801758]], [[26.642812728881836]], [[27.627708435058594]], [[31.602685928344727]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


class TestPrimitiveOp_2c9f49ffc83291f80720cf20c7de17b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5843.70654296875]], [[5937.93505859375]], [[5675.35400390625]], [[5809.3759765625]], [[5859.4814453125]], [[5682.23876953125]], [[5770.94189453125]], [[5707.72412109375]], [[5628.2763671875]], [[5966.88623046875]], [[6022.13232421875]], [[5729.32421875]], [[5904.7607421875]], [[5779.0244140625]], [[5859.49560546875]], [[5855.07177734375]], [[5661.494140625]], [[5929.25048828125]], [[5824.734375]], [[5685.28125]], [[6094.31201171875]], [[5796.33203125]], [[5748.5986328125]], [[5948.90283203125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7623a37850985b608478c32237dade99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[35929.70703125]], [[36433.05859375]], [[36965.9453125]], [[32376.0078125]], [[33316.94921875]], [[33616.84375]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_c5ac6b34de6f8e117c1a93828d211de4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6418.02197265625]], [[6568.80224609375]], [[6499.943359375]], [[6184.40185546875]], [[6656.0615234375]], [[6577.35107421875]], [[6668.833984375]], [[6889.89306640625]], [[6606.52880859375]], [[6713.30078125]], [[6550.7705078125]], [[6524.38427734375]], [[6736.95849609375]], [[6634.6806640625]], [[6608.33935546875]], [[6503.984375]], [[6628.8095703125]], [[6695.380859375]], [[6658.064453125]], [[6347.927734375]], [[6714.1513671875]], [[6508.87109375]], [[6294.87451171875]], [[6779.07177734375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e7b1138ff93bc36239dcbe2e3f4ad548(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[49677.06640625]], [[43435.1484375]], [[36693.43359375]], [[43158.00390625]], [[33826.00390625]], [[46592.015625]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_8de9e6655109cc38b75eb8e8dd0b13af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6884.412109375]], [[6921.4716796875]], [[6752.83447265625]], [[6797.185546875]], [[6787.24853515625]], [[7036.162109375]], [[6929.7431640625]], [[6748.119140625]], [[6988.8291015625]], [[6773.73046875]], [[6973.1845703125]], [[7152.58642578125]], [[6844.4853515625]], [[6926.77978515625]], [[6570.67578125]], [[6830.45556640625]], [[7039.6982421875]], [[6614.73681640625]], [[7105.6669921875]], [[6836.36962890625]], [[7082.83056640625]], [[6533.10205078125]], [[6986.79443359375]], [[6861.8408203125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba550ad7bcfe1bc75810f6523debac08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[41937.74609375]], [[48162.34375]], [[47176.3203125]], [[36864.703125]], [[39939.66796875]], [[44321.77734375]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_aef90684f360ca53e78aff3b59c6e6ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7660.421875]], [[7258.74951171875]], [[7272.490234375]], [[7295.19140625]], [[7042.4716796875]], [[7495.66015625]], [[6970.814453125]], [[6803.47802734375]], [[7211.8740234375]], [[7198.03369140625]], [[6839.02099609375]], [[6886.09716796875]], [[7275.51953125]], [[7244.890625]], [[7411.26513671875]], [[7317.38037109375]], [[7201.0263671875]], [[7134.201171875]], [[6972.4033203125]], [[7027.31103515625]], [[6857.54345703125]], [[7229.5224609375]], [[7435.80078125]], [[7020.15283203125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_523bba810dacf200c3b2066b7196547e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[36748.0625]], [[42465.46484375]], [[49356.62890625]], [[37389.99609375]], [[48896.0390625]], [[50524.62890625]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


class TestPrimitiveOp_a88e2ffd22d858d8427d312c20c0d8e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_defe42a6906c6b7513f92de86978631f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.0593037605285645]], [[5.837646961212158]], [[5.186929225921631]], [[5.490591049194336]], [[5.4441752433776855]], [[5.037900924682617]], [[5.012334823608398]], [[5.770382881164551]], [[5.6755781173706055]], [[5.36258602142334]], [[5.038281440734863]], [[5.764506816864014]], [[5.904150009155273]], [[5.765399932861328]], [[5.385991096496582]], [[5.383942127227783]], [[6.328894138336182]], [[5.334985256195068]], [[5.737000942230225]], [[5.298100471496582]], [[4.8958845138549805]], [[5.617877960205078]], [[5.112884521484375]], [[5.16455602645874]]]], dtype='float32').reshape([1, 24, 1, 1]),
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