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
        return True
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



class PrimitiveOp_a8ce5b019ecea14eeab1e8821734641b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2774c7541978089979a8bf75d9bb915d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8ce5b019ecea14eeab1e8821734641b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_15e9618881bbcf2704b6c3e42456f235(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b21e10151f5f42299c2f2e8d65245bf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b21e10151f5f42299c2f2e8d65245bf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21504, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_273f5fa5e1a60bc6bab9fc18b1756fdd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 576, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 576, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f8595731d0f004149e22ff848f5ed98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_273f5fa5e1a60bc6bab9fc18b1756fdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_45ed6e2ea6597cb8a974ca3264305339(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7285593bc04cb6b369b69c524e983c8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45ed6e2ea6597cb8a974ca3264305339
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_38bb39d657dc5d7e436622670defaaf4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 92, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_61bf352a65feab513291922baf1f3d7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38bb39d657dc5d7e436622670defaaf4
    def get_inputs(self):
        return [
            paddle.uniform([1, 92, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_365076adc3d102e71c719de9ef08dd8f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_63e40356286180d9f9b4c652500eb132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365076adc3d102e71c719de9ef08dd8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 152, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4f7308ce4f71bc6b95b05f00e639e2fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 160, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9bc44b61fc9876453d735eabd3109e21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7308ce4f71bc6b95b05f00e639e2fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_379f999eb4bce558d918c04cb94da334(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6cc6996d5711e48fa5edf70f433a34c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_379f999eb4bce558d918c04cb94da334
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class PrimitiveOp_fad09e57f2687163480f6dd7e30c1b7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 768, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d4a3a85ce228a85013cca5c194dfddb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fad09e57f2687163480f6dd7e30c1b7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f222e4c5dee0ef477fb91950adcd45f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e0daf7ae547bfbfe5390f925ada64f6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f222e4c5dee0ef477fb91950adcd45f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8570defc04449c14d5bee6ebca4fed58(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f22e76f2ea13917fef17230001a8e1f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8570defc04449c14d5bee6ebca4fed58
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aed5bda16e8cfde36cccb444f43454e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6b3be2b89331fbd84aec78dc35d6e2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aed5bda16e8cfde36cccb444f43454e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8256110661c5365bde62ac806223bfb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_379f999eb4bce558d918c04cb94da334
    def get_inputs(self):
        return [
            paddle.uniform([43, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c288f9c525290f5066f558941992718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.40505823493003845], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_adad8a01dc9927f8bd2cc2a7665e748b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8570defc04449c14d5bee6ebca4fed58
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a27c74de460ece435090f3e7f32c4d9e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e07043bdf5d883ed57fefd3ffd6b75c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a27c74de460ece435090f3e7f32c4d9e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_59945b092cb4d746030eeb6abfa865f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ba2a59bb137edb21273af43257adc078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba2a59bb137edb21273af43257adc078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba2a59bb137edb21273af43257adc078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a1b8001c62c0bd69cf5de364d5b27a34(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 336, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e7c5f13910a637aec8d4b04f848f10ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1b8001c62c0bd69cf5de364d5b27a34
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0aaf2290134d34f1cb502ed2c21714d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_3605aca3e677045f208df2206e91238b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_01421f7dff29dbb58036232da4159fa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class PrimitiveOp_4b25df3b1fffbc0b25118bd310e55451(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4424d65aefd462e0a2b2b009ff5eefa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b25df3b1fffbc0b25118bd310e55451
    def get_inputs(self):
        return [
            paddle.uniform([4, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.49516209959983826]], [[0.15384544432163239]], [[0.10794699937105179]], [[0.08243642002344131]]], dtype='float32').reshape([4, 1, 1]),
        ]


class PrimitiveOp_3b88e84456df8cb584439449ae567680(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 960, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6ca5f84180fed38620928baf4f302fca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b88e84456df8cb584439449ae567680
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_13ed9f40856a970d54ffe24723e19107(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2de1d525819d23857f870af61b6c9e01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed9f40856a970d54ffe24723e19107
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_476e6620e1438088f38ad912418c2905(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6cc05e8ee866ced8824fa896b7b68d7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bc7f49a71d68be5f48093de4be1538c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 1, 9, 112, 112], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 2, 16, 9, 112, 112], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef795d85fc8ac051a96cd676c3c3f52b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc7f49a71d68be5f48093de4be1538c8
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 1, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_94964cae99f5e12c43a76654feaef906(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a27c74de460ece435090f3e7f32c4d9e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e44d712d7a239b739f36a287393eb44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2483144998550415], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7bdfc8d95d5881032d537aaa3b422cec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7bdfc8d95d5881032d537aaa3b422cec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7bdfc8d95d5881032d537aaa3b422cec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0b73d20cebb10645d031e52e745d0cf6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 20, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d095d758acfbe11b141cbc3d8f3336aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b73d20cebb10645d031e52e745d0cf6
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.9306114315986633]], [[0.9031740427017212]], [[0.8208832144737244]], [[0.8823843002319336]], [[0.8743228316307068]], [[0.9052762985229492]], [[0.8928773999214172]], [[0.9185364842414856]], [[0.8435086011886597]], [[0.9584557414054871]], [[0.9349470138549805]], [[0.8886660933494568]], [[0.9429078698158264]], [[0.9535377025604248]], [[0.9543765187263489]], [[0.9342396855354309]], [[0.8669385313987732]], [[0.943714439868927]], [[0.9526184797286987]], [[0.9724346399307251]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class PrimitiveOp_9984ef09cd593fb4560d8d5176e4c7bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6770c548d1849d859d5eec9a24f05381(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9984ef09cd593fb4560d8d5176e4c7bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_42e5cbf8636ab2328cb24d25e02d55a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59bc67bb8c4068adc28e7888053442dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42e5cbf8636ab2328cb24d25e02d55a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a765bc0a04d60f18d97a972284ed0241(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 960, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6ddcdaf878e3026f546c21ed5819f0b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a765bc0a04d60f18d97a972284ed0241
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_eeacd4c721523530e52c46b66de64a38(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9ab6bcb6cca4fa418422559d1ec92366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72e4c9ecb974002ee15b0800eb6246ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72e4c9ecb974002ee15b0800eb6246ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_98420ee273a186f34c945b66bbab6eef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2fc02c1f53880cc76b297ae86ebc0876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98420ee273a186f34c945b66bbab6eef
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.23666216433048248]]], dtype='float32').reshape([1, 1, 1]),
        ]


class PrimitiveOp_85baf4be5ec839e5a123fc61e087f547(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2100, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_971db95433b2eb3c3823acc861468ed3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85baf4be5ec839e5a123fc61e087f547
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2db602b7bdcc61b7fce8fde72adb22df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.08490084111690521], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c94e0eeae3a50f039ec474da3d8e43a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_39be55716b940d949c945071172b4740(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9d35808d05c7e718499cbd4315f63d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39be55716b940d949c945071172b4740
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f35fb909862ead27f3bdb6714839bc40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 60, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ed32ec80b79f706aa05e171d48d76a58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f35fb909862ead27f3bdb6714839bc40
    def get_inputs(self):
        return [
            paddle.uniform([10, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3f65c315ee5b1f776a3c66ea436faaeb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_28119637b5e500ed0719d28e2a79da29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f65c315ee5b1f776a3c66ea436faaeb
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_289da9e370b87b8230fefc86aef88388(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_289da9e370b87b8230fefc86aef88388(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_289da9e370b87b8230fefc86aef88388(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_289da9e370b87b8230fefc86aef88388(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a7839bc29fa408f2645a06ed8d01664(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8570defc04449c14d5bee6ebca4fed58
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3c7a027b4b4f54f78609f64a7f9adc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3c7a027b4b4f54f78609f64a7f9adc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3c7a027b4b4f54f78609f64a7f9adc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70c553538f080589ff9e98db3bff902f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed9f40856a970d54ffe24723e19107
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c011a20f722f6fc4bbbbad775677bd7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a765bc0a04d60f18d97a972284ed0241
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_88a18fec0ab3d35435862b899ebaa447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39be55716b940d949c945071172b4740
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a3e4323a567951a3d707091b7981614b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1152, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1152, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe08d16f366db21b0fa73f21bb7dac1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3e4323a567951a3d707091b7981614b
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6c5233add8cdd8c5d32d11ce72b27a52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f291d38a1bfa6c02b08749a03694f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c5233add8cdd8c5d32d11ce72b27a52
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed18e33ab9f3afd2191f1914a2b7136c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42e5cbf8636ab2328cb24d25e02d55a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_43b3bb766f0fe3e3d72d9373a8e2b643(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f65c315ee5b1f776a3c66ea436faaeb
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a0cd5fa6582c6fa47ad19ded5110ad5e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 68, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8da5aa9b3dbc74f132e3a1de03f49426(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0cd5fa6582c6fa47ad19ded5110ad5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17064687609672546], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_77a23f49cc975ba1ce24c8b8287d5231(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0a28c1598c7bd0e66ecc7f7542934b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_40ca935aa07f787a0709f3b9b31f66a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1b8001c62c0bd69cf5de364d5b27a34
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c425c3d49b25cb863b2948c5dafb2fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c5233add8cdd8c5d32d11ce72b27a52
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e8478ea0b968c8d40ce8f722cc20497(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.44347327947616577], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c854e9b7faf7ee1767f48ca3ec4c53fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c5233add8cdd8c5d32d11ce72b27a52
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_93f34d002b99b46d05d1eb89caaa868d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_681d37aa33056a0ab817d03c3a51b4ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93f34d002b99b46d05d1eb89caaa868d
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.9458035230636597]], [[0.9639922976493835]], [[0.864982008934021]], [[0.9046134352684021]], [[0.766802191734314]], [[0.8668656349182129]], [[0.6934769153594971]], [[0.6689972281455994]], [[0.94728684425354]], [[0.7830797433853149]], [[0.7823899984359741]], [[0.9436419606208801]], [[0.8246046304702759]], [[0.7450451850891113]], [[0.8535295128822327]], [[0.8245354294776917]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_0beff2f0d141cfff69baa908fd64f15c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c5233add8cdd8c5d32d11ce72b27a52
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_248307a896aa76b5b3c9b8b9872b5590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1b8001c62c0bd69cf5de364d5b27a34
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a69e43a060f8a1b0a2aac3060ad88535(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 44, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 44, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_659e644eb938f291314ef6e160410f10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a69e43a060f8a1b0a2aac3060ad88535
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7a67ea2ca29d795eebf27ef76bd63d15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7308ce4f71bc6b95b05f00e639e2fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_87588b7572fd527a107b0fe5e4d81470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.17793193459510803, 0.10853168368339539]], [[-0.0397617369890213, 0.13806255161762238]], [[-0.015750199556350708, -0.38073936104774475]], [[-0.04490005970001221, -0.03202305734157562]], [[0.3664105236530304, 0.11237730085849762]], [[0.2510986924171448, -0.059057846665382385]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_e9981d20101aa55a88a36187b41cf65a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[-0.22358153760433197, 0.18547365069389343]], [[-0.12822601199150085, 0.306474506855011]], [[-0.04643663763999939, -0.39308109879493713]], [[-0.16461046040058136, 0.07661989331245422]], [[0.42683181166648865, 0.023946627974510193]], [[-0.026828020811080933, 0.06228762865066528]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_cbe64e59b2f2d9aa9e19b32ccb58ef9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.17793193459510803, 0.10853168368339539]], [[-0.0397617369890213, 0.13806255161762238]], [[-0.015750199556350708, -0.38073936104774475]], [[-0.04490005970001221, -0.03202305734157562]], [[0.3664105236530304, 0.11237730085849762]], [[0.2510986924171448, -0.059057846665382385]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.17793193459510803, 0.10853168368339539]], [[-0.0397617369890213, 0.13806255161762238]], [[-0.015750199556350708, -0.38073936104774475]], [[-0.04490005970001221, -0.03202305734157562]], [[0.3664105236530304, 0.11237730085849762]], [[0.2510986924171448, -0.059057846665382385]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_eda2953204000d8cb52f6916d3062e56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.22358153760433197, 0.18547365069389343]], [[-0.12822601199150085, 0.306474506855011]], [[-0.04643663763999939, -0.39308109879493713]], [[-0.16461046040058136, 0.07661989331245422]], [[0.42683181166648865, 0.023946627974510193]], [[-0.026828020811080933, 0.06228762865066528]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[-0.22358153760433197, 0.18547365069389343]], [[-0.12822601199150085, 0.306474506855011]], [[-0.04643663763999939, -0.39308109879493713]], [[-0.16461046040058136, 0.07661989331245422]], [[0.42683181166648865, 0.023946627974510193]], [[-0.026828020811080933, 0.06228762865066528]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_7fe30f0d32b9907e1405e7f27734bb7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98420ee273a186f34c945b66bbab6eef
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.009053535759449005], [0.0029657601844519377], [0.055334657430648804], [0.00016773739480413496], [0.056294720619916916], [0.017163602635264397]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([[[0.1702578216791153], [0.36321955919265747], [0.33214816451072693], [0.18884220719337463], [0.14932842552661896], [0.21366457641124725]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_89cac59e400306bc2a994cc63c2c71d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98420ee273a186f34c945b66bbab6eef
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.024514904245734215], [0.036666370928287506], [0.06201189383864403], [0.0059858146123588085], [0.07812994718551636], [0.00031193543691188097]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([[[0.1702578216791153], [0.36321955919265747], [0.33214816451072693], [0.18884220719337463], [0.14932842552661896], [0.21366457641124725]]], dtype='float32').reshape([1, 6, 1]),
        ]


class PrimitiveOp_24220e97cfb882d7db6688d0b0dceb7a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5cb26ca917c992fc761401e0c3ff2c82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24220e97cfb882d7db6688d0b0dceb7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d9977247759bc092dedb9973d521f77e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 1, 49, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 4, 16, 49, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3ab87ce6092bcd301f05a261e1b353aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9977247759bc092dedb9973d521f77e
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 1, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06b196423ce79bcad89057ff84d5c7c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a765bc0a04d60f18d97a972284ed0241
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b75c3c39962f68793e377df24762be4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f65c315ee5b1f776a3c66ea436faaeb
    def get_inputs(self):
        return [
            paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_991aa3649a8d89ca46c1ce9d0743997d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8ce5b019ecea14eeab1e8821734641b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5419101c4f71e06d7120314964b7b1b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a27c74de460ece435090f3e7f32c4d9e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_069cfdf0c9713bfac471b083069991e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8570defc04449c14d5bee6ebca4fed58
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ef45dcd3ae0df98a8bfad37f110b9f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ef45dcd3ae0df98a8bfad37f110b9f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ef45dcd3ae0df98a8bfad37f110b9f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ef45dcd3ae0df98a8bfad37f110b9f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7c987637f3a77cd7027758e096b15014(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_379f999eb4bce558d918c04cb94da334
    def get_inputs(self):
        return [
            paddle.uniform([43, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e5de0e4ce8fa8364a392f3166e466ad4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 81], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0ef8bef711f8b10ab2b61274af46cfac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5de0e4ce8fa8364a392f3166e466ad4
    def get_inputs(self):
        return [
            paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d7352dd94434211fb041a08a13babfb1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_645b15abf4d8638ad0bf9d93346db577(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.1081929206848145, 2.0675156116485596, 2.0014145374298096, 2.2239789962768555, 2.3097641468048096, 2.0764319896698, 1.8632161617279053, 2.0325520038604736, 1.9944472312927246, 2.2200491428375244, 2.2384321689605713, 2.175208806991577, 2.0629239082336426, 2.0931596755981445, 2.230994701385498, 1.9065513610839844], dtype='float32').reshape([16]),
            paddle.to_tensor([0.8237231969833374, 0.9899024367332458, 0.7822033166885376, 0.6354637742042542, 0.6508806943893433, 0.9901790618896484, 0.567777156829834, 0.8425178527832031, 0.8346413373947144, 0.8164232969284058, 0.7743445634841919, 0.8362159729003906, 0.5368951559066772, 0.7441719770431519, 0.766444206237793, 0.6562821865081787], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_6f24811bf0d0212043da024d4b6b377c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.0443060398101807, 1.9601728916168213, 2.1959762573242188, 1.9368762969970703, 1.9029803276062012, 1.912628173828125, 2.331761121749878, 1.968799352645874, 2.0185184478759766, 2.0568387508392334, 2.1085896492004395, 1.994911789894104, 2.0172736644744873, 1.9499866962432861, 2.0252716541290283, 2.2711005210876465], dtype='float32').reshape([16]),
            paddle.to_tensor([0.17627683281898499, 0.010097586549818516, 0.2177966833114624, 0.36453622579574585, 0.34911927580833435, 0.009820925071835518, 0.432222843170166, 0.15748214721679688, 0.16535866260528564, 0.18357667326927185, 0.2256554216146469, 0.16378404200077057, 0.46310481429100037, 0.25582805275917053, 0.23355577886104584, 0.3437178134918213], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_c9e3bf4888203ffcb6d80b4a109267f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.524232804775238, 0.516607940196991, 0.5109473466873169, 0.5298299193382263, 0.541936993598938, 0.5187058448791504, 0.5164330005645752, 0.5056280493736267, 0.4996069073677063, 0.5475218892097473, 0.5522831082344055, 0.5364197492599487, 0.510445773601532, 0.5141330361366272, 0.5457367300987244, 0.5079633593559265], dtype='float32').reshape([16]),
            paddle.to_tensor([0.3236719071865082, 0.4011479318141937, 0.3258005380630493, 0.09456221014261246, 0.20205162465572357, 0.29172614216804504, 0.09821265935897827, 0.3831166923046112, 0.19772659242153168, 0.41344743967056274, 0.49105221033096313, 0.11943116039037704, 0.27239513397216797, 0.17290277779102325, 0.3967116177082062, 0.4324886202812195], dtype='float32').reshape([16]),
        ]


class PrimitiveOp_dba7f635df5585d7dd15cf6f26aa306e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7f6839d00f935d52405dd6bff2074ffa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dba7f635df5585d7dd15cf6f26aa306e
    def get_inputs(self):
        return [
            paddle.uniform([145, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d938e456cd4cb61f090757181d700253(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.34983187913894653], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_33e669bf74ae8c6199fad52a6cd08283(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4b241792bb25fe583a4802659145add7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33e669bf74ae8c6199fad52a6cd08283
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da79dd60741bbd1567e79af19c32f84c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da79dd60741bbd1567e79af19c32f84c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da79dd60741bbd1567e79af19c32f84c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5301cae3dcc9682f128ffa50bd3fe5f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4840018153190613], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_e6b115870dac2bd4a5c9de7ce7b479b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 300, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0a76267297e4e7366b561dd529b69889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b115870dac2bd4a5c9de7ce7b479b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.48930177092552185, 0.1727525293827057, 0.09296277910470963, 0.1347460299730301]]], dtype='float32').reshape([1, 1, 4]),
        ]


class PrimitiveOp_19c2ac00d767b97e264e68b3acd0e778(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4796ed25722efdf68443f1d9695ceac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19c2ac00d767b97e264e68b3acd0e778
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5580c17c9979fea4569c5ac24ecb4eb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.02037999778985977], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bc0a73faf09b1bb26758c112c2f2552d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.12783189117908478], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ecac63e8e9331296cf48b13c7d9950aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.05800976604223251], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ba2a59bb137edb21273af43257adc078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba2a59bb137edb21273af43257adc078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba2a59bb137edb21273af43257adc078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba2a59bb137edb21273af43257adc078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae180e04ef411d9166814961f037b343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae180e04ef411d9166814961f037b343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae180e04ef411d9166814961f037b343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dcfc564428cf84e56fbc415031654c70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8570defc04449c14d5bee6ebca4fed58
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_86a54ec7416ee4fcd7a5c436627ddd16(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8ed937aa30e265629479823e3130929(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86a54ec7416ee4fcd7a5c436627ddd16
    def get_inputs(self):
        return [
            paddle.uniform([3, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.47794291377067566]], [[0.3767252564430237]], [[0.2085568755865097]]], dtype='float32').reshape([3, 1, 1]),
        ]


class PrimitiveOp_6eb4a6e9a3ff24c70e63d79aecc77097(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_44da7c5b1df417b4f46a65213634a9a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6eb4a6e9a3ff24c70e63d79aecc77097
    def get_inputs(self):
        return [
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b927250f1ed493ebd521e2f1cd143a6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db8ac11661fe8ba5fbddcd902fc85009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f35fb909862ead27f3bdb6714839bc40
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f22e76f2ea13917fef17230001a8e1f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8570defc04449c14d5bee6ebca4fed58
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_262e8e2b50b2e6211b88a8a88dfc702f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0cd5fa6582c6fa47ad19ded5110ad5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.06869208067655563], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_495d91e33a635b19edea897af6cd1867(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c2dd8865c749aece8a06806458645bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_495d91e33a635b19edea897af6cd1867
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b83079c8e510360c3f05b8bea7211ff4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8ce5b019ecea14eeab1e8821734641b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_046bf242e687e1d6da8d5dea3eaa45d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a27c74de460ece435090f3e7f32c4d9e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6cf52968ff726dfa7ab970ab9887cee8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 320, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 320, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ff6c0b867464b3e360c597f7a4122a85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cf52968ff726dfa7ab970ab9887cee8
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd97aa78fd5322c7b85710e4b6f0eb2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a1ca419767c56c49909af1ef11eb453c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 872, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a9314b938952134be5cecdd63415564b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1ca419767c56c49909af1ef11eb453c
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5868663be13fae4fb58dbb4084b94ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5868663be13fae4fb58dbb4084b94ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5868663be13fae4fb58dbb4084b94ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5868663be13fae4fb58dbb4084b94ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2f3bc8fd07c5dacb5e606a94f7719f18(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 100, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_314d180723704f74ed92488886ff4183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f3bc8fd07c5dacb5e606a94f7719f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52a959cf135117882a736b88e937bf24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a765bc0a04d60f18d97a972284ed0241
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_efb9086f821a20072586613485e9cf86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f241508e772a06eaa1cdab4834e4f76e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f241508e772a06eaa1cdab4834e4f76e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f241508e772a06eaa1cdab4834e4f76e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f241508e772a06eaa1cdab4834e4f76e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f241508e772a06eaa1cdab4834e4f76e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8255bf32926a0c875bc5d96ec5432240(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8255bf32926a0c875bc5d96ec5432240(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f241508e772a06eaa1cdab4834e4f76e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1fe14a1ada2bb144a12326ab62490cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72a5201f5149e416eff361ebf9d70872(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_379f999eb4bce558d918c04cb94da334
    def get_inputs(self):
        return [
            paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_79a948b04e333568a460bcf0044d11cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3034917414188385], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5e62cb016eb141b2b79cb445be02b533(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8570defc04449c14d5bee6ebca4fed58
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_691fafab64d3fa33d0c13d528a8a6f2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_379f999eb4bce558d918c04cb94da334
    def get_inputs(self):
        return [
            paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class PrimitiveOp_de41dd47297a46f2044aa6bf6d6b5a35(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bbcd99e7a0f19a67a3034a0962b80b51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de41dd47297a46f2044aa6bf6d6b5a35
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0e3a2e8016af8286bb2272e23688618(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0cd5fa6582c6fa47ad19ded5110ad5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.41016507148742676], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f3da1e67cebce87064c8d2b0a9e3fef2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 4, 1, 49, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 4, 16, 49, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de23b4cd222e5c7f85458629b4e37119(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3da1e67cebce87064c8d2b0a9e3fef2
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 1, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f0050d3cbc1aef0b73f7bc36c6e49a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f0050d3cbc1aef0b73f7bc36c6e49a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f0050d3cbc1aef0b73f7bc36c6e49a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c700c32d13ac44b6a0488c6b78025c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c700c32d13ac44b6a0488c6b78025c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c700c32d13ac44b6a0488c6b78025c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c700c32d13ac44b6a0488c6b78025c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b018881c7b0a50025013fbfa32434287(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42e5cbf8636ab2328cb24d25e02d55a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_81532de63423082f34c4e132cb97f22e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1b8001c62c0bd69cf5de364d5b27a34
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7a18c346cd091828fece527d0a42ec6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39be55716b940d949c945071172b4740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3fc6489618b07c3077d4bbc369be5dba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_379f999eb4bce558d918c04cb94da334
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6558a47673440af5766878b002e3a19c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1206e5b0a8c8146e25b7be8c0133603b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6558a47673440af5766878b002e3a19c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e648d405f232958c2e8a605a38f98a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8570defc04449c14d5bee6ebca4fed58
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0bb5334aa97663b65be2fbd12fcab3eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cf52968ff726dfa7ab970ab9887cee8
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bf3088cc16f31c965831062ca57f3f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bf3088cc16f31c965831062ca57f3f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bf3088cc16f31c965831062ca57f3f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af869ff0a49f274c4820852bdd8e359b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 96, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_173b4141e6caed499d3669b0df217ade(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8ce5b019ecea14eeab1e8821734641b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d60aaf967df7afd00148d2c6a03a623(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8570defc04449c14d5bee6ebca4fed58
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4a2605b3435a2b4b649f4a5cfc03b38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4a2605b3435a2b4b649f4a5cfc03b38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4a2605b3435a2b4b649f4a5cfc03b38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4a2605b3435a2b4b649f4a5cfc03b38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_75ab1ce1ca742268bc1923ad626c28f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_81bdaba4544131da9393093343b5ceec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75ab1ce1ca742268bc1923ad626c28f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0efa05647d72a5e5ba200c07fc224dc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.1107863038778305], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_79d78c488f3aabc906b9a8f56888a696(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0029904842376708984], [-0.25137341022491455], [-0.3327561616897583], [-0.10126486420631409], [-0.27697935700416565], [0.1574983447790146], [0.10954436659812927], [-0.2752947509288788], [-0.032525770366191864]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.12475007772445679], [-0.004233710467815399], [0.2707362771034241], [0.04699438810348511], [0.15774263441562653], [0.42633551359176636], [-0.09890206158161163], [-0.2835497558116913], [-0.4208599328994751]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_3da6c8c6f8bafefedd9b2f3bb19d5437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20529648661613464], [-0.18322747945785522], [0.36298295855522156], [-0.24767747521400452], [0.0681171715259552], [0.2796595096588135], [-0.06590437889099121], [0.13182024657726288], [-0.03437986969947815]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.29441577196121216], [-0.21231204271316528], [-0.11227650940418243], [-0.09754018485546112], [-0.15886008739471436], [-0.0820966362953186], [-0.12468622624874115], [0.20683139562606812], [-0.06767536699771881]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_53c4b8c05b52b6149882da5bbeab4639(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2882349491119385], [-0.06873087584972382], [0.36298295855522156], [-0.10126486420631409], [0.13070568442344666], [0.32637155055999756], [0.18734493851661682], [0.13182024657726288], [0.2707979679107666]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.12475007772445679], [0.10680622607469559], [0.2707362771034241], [0.3789668083190918], [0.2584306001663208], [0.42633551359176636], [-0.09890206158161163], [0.20683139562606812], [-0.06767536699771881]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_5eef900d2786624d1fc326e79093a971(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed9f40856a970d54ffe24723e19107
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c96a3c6c52cec6689d26fdc752bb6a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dba7f635df5585d7dd15cf6f26aa306e
    def get_inputs(self):
        return [
            paddle.uniform([10, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2d25e44fcd5823b98187b408f9d56b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe743c2606a69acd1286ac7b4b988e8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a8db7d79ba603928916a4aab0dfca6a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_273f5fa5e1a60bc6bab9fc18b1756fdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b679c4cc71d59f2e43b2dfcd94557384(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9fca25ed6b0440a705671cef1b94422e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b679c4cc71d59f2e43b2dfcd94557384
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4245343804359436], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9fca25ed6b0440a705671cef1b94422e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b679c4cc71d59f2e43b2dfcd94557384
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4245343804359436], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7fe80bf45124ca8664ea8278252b5644(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a8f0a74aba79956fd7249534ee6f6a74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fe80bf45124ca8664ea8278252b5644
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e729ce4a58e1fb04f064fdd49605571c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cd211251176b8ac1dde51cc0f897e910(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e729ce4a58e1fb04f064fdd49605571c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.03257932513952255], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_957fba86d9207bcce9276ec0f803dfb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fad09e57f2687163480f6dd7e30c1b7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9db9c0bedcb79d1b3e7b957f664d4e3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_273f5fa5e1a60bc6bab9fc18b1756fdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2fc77f67d6cf0dc520abb3c5d7d1a0a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8ce5b019ecea14eeab1e8821734641b
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dce2e127fb63a8c0220c35d4995a198a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dce2e127fb63a8c0220c35d4995a198a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dce2e127fb63a8c0220c35d4995a198a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dce2e127fb63a8c0220c35d4995a198a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dce2e127fb63a8c0220c35d4995a198a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0920a4d621ae5f24d2aedd627360369(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([5504, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0920a4d621ae5f24d2aedd627360369(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([5504, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dce2e127fb63a8c0220c35d4995a198a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5504, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_084821d92360c14e4e4d0a29cf72c48f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ae11320503813de712beff0767170a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_084821d92360c14e4e4d0a29cf72c48f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_933ea103ec7225dd1e46ab506d5626c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_273f5fa5e1a60bc6bab9fc18b1756fdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c363809de22c61b93144f162a2acac3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_379f999eb4bce558d918c04cb94da334
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c07474e27d158ce947768c05ff5320e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c07474e27d158ce947768c05ff5320e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c07474e27d158ce947768c05ff5320e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe1c253e39d4306f9595928e91e744cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42e5cbf8636ab2328cb24d25e02d55a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_401412e677d3b048518a6d278330ab29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.31402257084846497], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ec9bfc40b8420156251b1946c26a8198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8570defc04449c14d5bee6ebca4fed58
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b2f8d98762f1c3a2cf490994053eea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39be55716b940d949c945071172b4740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d6ace31f971fd90e501e4112037e132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed9f40856a970d54ffe24723e19107
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_373d8a0ec1c910abd02992266543699b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1b8001c62c0bd69cf5de364d5b27a34
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32d5f3eafc0cb9283e7e2c05b339c2f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e30ed52cb5ae3242d681ad560a32d658(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6a2c2b69682293c75df0cb693e3cd442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30ed52cb5ae3242d681ad560a32d658
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_61c8b51cb4ec38acb1b96553571d8877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed9f40856a970d54ffe24723e19107
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ccbde913955e28ec5f8831477dc3555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5de0e4ce8fa8364a392f3166e466ad4
    def get_inputs(self):
        return [
            paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6b2f8e9b3c562537f4a502427e327a01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365076adc3d102e71c719de9ef08dd8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e30c9929f70d7657a2c5aff0a3dc5a0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38ca9bd1c920c14943ba277d29c86fbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c5233add8cdd8c5d32d11ce72b27a52
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a5bd45f699884012df82bedd7c8f9922(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.24855555593967438], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_36e122a8b324ef1f39afe76c04516af1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f65c315ee5b1f776a3c66ea436faaeb
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0ddef7649e40f1f60b4c016c1665f9cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8806150555610657], dtype='float32').reshape([1]),
            paddle.to_tensor([0.3223762810230255], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4332eea9a01a1136d8b5e4a09d5eb288(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.804711103439331], dtype='float32').reshape([1]),
            paddle.to_tensor([0.12384786456823349], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_391ea07d377bd44a525315beafa2d798(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 32, 1, 49, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 32, 16, 49, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_790f78bb904d1b4385831401401b9ae3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_391ea07d377bd44a525315beafa2d798
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_077b080259d38ef4229744f4eb9ffabd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_700f00bcfbbb1f4307af62e00093f8c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_077b080259d38ef4229744f4eb9ffabd
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_318112fbb3a78cef01b64ba6ced6256d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbd38b8ca90f7c41e51c4ec40edf9af5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed9f40856a970d54ffe24723e19107
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d8062088484c8a20e453b9da3d59d47c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a27c74de460ece435090f3e7f32c4d9e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 168, 168], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79acfb528c3c78f029a791d8e7343f35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c5233add8cdd8c5d32d11ce72b27a52
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_db95b78c01a0cd6760b50ec1358a26e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 36, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b88a37be84085cefdbda1eb7ea7a6e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db95b78c01a0cd6760b50ec1358a26e7
    def get_inputs(self):
        return [
            paddle.uniform([10, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1be5fb693a02140984bf12997f7258f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0cd5fa6582c6fa47ad19ded5110ad5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1991686075925827], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_88a18fec0ab3d35435862b899ebaa447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39be55716b940d949c945071172b4740
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_719611a5f13c0bd0c2cb9ce0c0f12abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.35294264554977417], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3a45f0286abd0a8d30448cb98eae1a98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c5233add8cdd8c5d32d11ce72b27a52
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_81bdaba4544131da9393093343b5ceec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75ab1ce1ca742268bc1923ad626c28f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae3d6dc9230440962a951bcc6cf55409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_077b080259d38ef4229744f4eb9ffabd
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e45eaebc956fdf8d427bedc5cdaef6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8570defc04449c14d5bee6ebca4fed58
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e949a5657ad654e5702c982d32c59227(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.0958491712808609, -0.25636279582977295, -0.001880258321762085, 0.060361623764038086, -0.07491792738437653, 0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.372331440448761, -0.2269829362630844, -0.3053029179573059, -0.09710273146629333, -0.08524835109710693, -0.2023477703332901], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_01d686019637336745bd47b47c187959(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.035687658935785294, 0.05818998068571091, 0.000574048375710845, -0.005861278623342514, 0.006386629771441221, -0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_1549769b68e47a22d4f772c8bbf24135(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 0.0, 0.0, -0.005861278623342514, 0.0, -0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_0d13679ab4ef4c4409a8537dac5a6c18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.023094341158866882, 0.0, 0.0, 0.09965561330318451, 0.24972057342529297, 0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.06807205080986023, 0.27504706382751465], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_4a905939ca3166e9381749fbe1e0964d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20614641904830933, -0.25636279582977295, 0.07439911365509033, 0.12330588698387146, -0.07491792738437653, 0.22616015374660492], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.2906482219696045, -0.13422895967960358, -0.13182006776332855, -0.0787929892539978, 0.1125604510307312, -0.2023477703332901], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_da306646cb112164407dfd98d8ce7ade(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.2104695439338684, 0.10339093208312988, -0.06354057788848877, -0.0511191189289093, -0.07478326559066772, -0.06795307993888855], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.2104695439338684, 0.10339093208312988, -0.06354057788848877, -0.0511191189289093, -0.07478326559066772, -0.06795307993888855], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_d277557f0b80fac05e464197db7d3e62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20148581266403198, -0.19614319503307343, 0.1412818729877472, -0.20846062898635864, -0.17556460201740265, -0.1429460048675537], dtype='float32').reshape([6]),
            paddle.to_tensor([0.20148581266403198, -0.19614319503307343, 0.1412818729877472, -0.20846062898635864, -0.17556460201740265, -0.1429460048675537], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_41743815d526d303123274d03eda4b79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3250899314880371, 0.0, 0.07627937197685242, 0.16259987652301788, 0.24972057342529297, 0.22616015374660492], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3250899314880371, 0.0, 0.07627937197685242, 0.16259987652301788, 0.24972057342529297, 0.22616015374660492], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_473f1d2441f51b84d94464baec7eed54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08168324828147888, 0.09275397658348083, 0.17348286509513855, 0.018309742212295532, 0.26588085293769836, 0.27504706382751465], dtype='float32').reshape([6]),
            paddle.to_tensor([0.08168324828147888, 0.09275397658348083, 0.17348286509513855, 0.018309742212295532, 0.26588085293769836, 0.27504706382751465], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_1d8928b4087bf30de7f1e91c56622378(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.07782649993896484, 0.3221953511238098, -0.5356611013412476, -0.2767977714538574, -0.766766369342804, 0.00524621969088912], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.19202905893325806, 0.7949846386909485, -1.3216899633407593, -0.6829706430435181, -1.8919188976287842, 0.012944519519805908], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_8cfcd81ee639b044f10f89098bb42e54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.01472488697618246, 0.20391061902046204, 0.4145123362541199, 0.15898877382278442, 0.5919466018676758, 6.790518091293052e-05], dtype='float32').reshape([6]),
            paddle.to_tensor([0.014944949187338352, 0.2561403512954712, 0.7079778909683228, 0.1890447586774826, 1.4506597518920898, 6.790979387005791e-05], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ca39728c53ea119b05babd7a5eabc1df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aed5bda16e8cfde36cccb444f43454e4
    def get_inputs(self):
        return [
            paddle.uniform([10, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ab9aa7a7fba56756f211670a4d29dad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_495d91e33a635b19edea897af6cd1867
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57eecf821a68fe7ca9cb91228e491a4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_077b080259d38ef4229744f4eb9ffabd
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07e4561a743ad7f13ee090bb53532ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07e4561a743ad7f13ee090bb53532ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07e4561a743ad7f13ee090bb53532ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07e4561a743ad7f13ee090bb53532ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07e4561a743ad7f13ee090bb53532ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a15328330314d6b2a3605efa84e9407(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1811, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a15328330314d6b2a3605efa84e9407(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1811, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07e4561a743ad7f13ee090bb53532ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1811, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2fc77f67d6cf0dc520abb3c5d7d1a0a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8ce5b019ecea14eeab1e8821734641b
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_68c48677c7bbaf3489b1be408a6d1c47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c5731bf660ecd11e044f32436b14163a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68c48677c7bbaf3489b1be408a6d1c47
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fdca225a0558864bff16f123528a0fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68c48677c7bbaf3489b1be408a6d1c47
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_504ec092db58f4046f7ca852afc2bc7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1b8001c62c0bd69cf5de364d5b27a34
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1fe14a1ada2bb144a12326ab62490cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_019a428061f24ae4abcedf66d37b7dd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3e4323a567951a3d707091b7981614b
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d8e6a112cd2cd8cc73cf9ec2a91dd8b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82d5ad02c80626cfc808b7011962500e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.07877841591835022], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_91aa37acc2c39060693c7c183cd1c99d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_29011776bb501c2a7449822400f1dc74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91aa37acc2c39060693c7c183cd1c99d
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d2e1ee0b892f8c5e529fbb3f74d1d49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42e5cbf8636ab2328cb24d25e02d55a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8da80925aa2163f3ebf62d9bbdff67ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75ab1ce1ca742268bc1923ad626c28f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1adf363743d54762fc72832310dd3db9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fad09e57f2687163480f6dd7e30c1b7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_801689fb2d7b4652adc7dd2a018767a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dba7f635df5585d7dd15cf6f26aa306e
    def get_inputs(self):
        return [
            paddle.uniform([171, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13b0442a6b79f982d641b8d61db49d9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1b8001c62c0bd69cf5de364d5b27a34
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e899c64cf254931f2d3bdac698e40af6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8ce5b019ecea14eeab1e8821734641b
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e148209ad3a18dfd71f439ef3cb4cd1f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b1fa3c6df75f3d903bedbb466b7f050(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e148209ad3a18dfd71f439ef3cb4cd1f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd2966c8536acb6d9946196707b68c8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8570defc04449c14d5bee6ebca4fed58
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a915a92a6ce8c037d7cd3aa347042205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18368826806545258], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_64b3661e08a9412b02da4daf9ca243b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 8, 1, 49, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 8, 16, 49, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_36f01420ed2327f7cf7f3850d440b720(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64b3661e08a9412b02da4daf9ca243b1
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3d9ea195305c2425380b1562549b935(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.0661697387695312, 1.9399363994598389, 2.1428277492523193, 2.1048550605773926, 2.0073599815368652, 2.2567760944366455, 2.117408037185669, 2.249511957168579, 2.203714609146118, 2.2638559341430664, 1.9596232175827026, 2.1692466735839844, 2.1125130653381348, 1.9698776006698608, 1.8564378023147583, 1.9076261520385742, 2.3275105953216553, 2.171522855758667, 1.9673293828964233, 2.1681442260742188, 2.091169834136963, 2.1324877738952637, 2.0330145359039307, 2.1752407550811768], dtype='float32').reshape([24]),
            paddle.to_tensor([0.8828868865966797, 0.6255854368209839, 0.7607230544090271, 0.7841118574142456, 0.7453173398971558, 0.8564379215240479, 0.6672857999801636, 0.8867545127868652, 0.5618027448654175, 0.8459969758987427, 0.7364556789398193, 0.5263252258300781, 0.8677802681922913, 0.9961231350898743, 0.557140588760376, 0.6060033440589905, 0.8514354228973389, 0.8514597415924072, 0.653854489326477, 0.8407319188117981, 0.5382512211799622, 0.9203518629074097, 0.6213589906692505, 0.8471647500991821], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_9a73520d06493846502ac9dbbb7d0e14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.188444137573242, 2.1940958499908447, 2.0810275077819824, 2.2405285835266113, 2.351086139678955, 1.9158568382263184, 2.007117986679077, 2.1231837272644043, 2.2281181812286377, 1.869976282119751, 2.1209330558776855, 1.962336540222168, 2.0042619705200195, 1.9317293167114258, 1.8772971630096436, 2.359963893890381, 1.9822226762771606, 2.0440967082977295, 2.1273930072784424, 2.2557408809661865, 2.1456761360168457, 1.9838593006134033, 1.9923783540725708, 2.333921432495117], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1171131432056427, 0.3744145631790161, 0.2392769455909729, 0.2158881276845932, 0.25468266010284424, 0.14356206357479095, 0.33271417021751404, 0.11324551701545715, 0.4381972551345825, 0.1540030539035797, 0.2635442912578583, 0.4736747741699219, 0.13221971690654755, 0.003876861184835434, 0.442859411239624, 0.3939966559410095, 0.14856456220149994, 0.14854028820991516, 0.34614554047584534, 0.1592680811882019, 0.46174877882003784, 0.07964815944433212, 0.3786410093307495, 0.15283527970314026], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_a87aba5deba4ad911eca8c30f9ec3058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5201224088668823, 0.5087743401527405, 0.5320100784301758, 0.5335363149642944, 0.5237252712249756, 0.5519582629203796, 0.5201781988143921, 0.5588014721870422, 0.5536020398139954, 0.5507993102073669, 0.5005338788032532, 0.5178096294403076, 0.5245500206947327, 0.49243244528770447, 0.46641889214515686, 0.5214614272117615, 0.5690532326698303, 0.5381487607955933, 0.5056836605072021, 0.5455238819122314, 0.5290845036506653, 0.5301624536514282, 0.5044069886207581, 0.5498732328414917], dtype='float32').reshape([24]),
            paddle.to_tensor([0.32896944880485535, 0.41770875453948975, 0.23582826554775238, 0.48949897289276123, 0.12923763692378998, 0.1925070434808731, 0.03956484794616699, 0.2958824932575226, 0.16539449989795685, 0.007382487878203392, 0.11560265719890594, 0.3615935742855072, 0.39369478821754456, 0.08359850198030472, 0.290140300989151, 0.32681742310523987, 0.1630883365869522, 0.4730071723461151, 0.3645085096359253, 0.4340798258781433, 0.31143465638160706, 0.03729693964123726, 0.44449102878570557, 0.027625858783721924], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_ac87c780d6f6cd248f9a89cadba48259(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f35fb909862ead27f3bdb6714839bc40
    def get_inputs(self):
        return [
            paddle.uniform([171, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0aaf2290134d34f1cb502ed2c21714d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_3605aca3e677045f208df2206e91238b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_01421f7dff29dbb58036232da4159fa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_a3c7a027b4b4f54f78609f64a7f9adc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3c7a027b4b4f54f78609f64a7f9adc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3c7a027b4b4f54f78609f64a7f9adc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3c7a027b4b4f54f78609f64a7f9adc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c700c32d13ac44b6a0488c6b78025c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c700c32d13ac44b6a0488c6b78025c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c700c32d13ac44b6a0488c6b78025c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_62ab1e559da9d97e83cd8cb98818aecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_62ab1e559da9d97e83cd8cb98818aecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_62ab1e559da9d97e83cd8cb98818aecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_62ab1e559da9d97e83cd8cb98818aecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_62ab1e559da9d97e83cd8cb98818aecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e1a3961c2c320fa1278f18418deaa85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1559, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e1a3961c2c320fa1278f18418deaa85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1559, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_62ab1e559da9d97e83cd8cb98818aecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1559, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9794fdf6fb3d3d5ff6d5c5f8cd521f6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9794fdf6fb3d3d5ff6d5c5f8cd521f6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_001c4f48668c9a6460435f09dd8fab91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98420ee273a186f34c945b66bbab6eef
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24512165784835815], [0.24658147990703583]]], dtype='float32').reshape([1, 2, 1]),
        ]


class PrimitiveOp_f3fc74df3abfedf16441aca8fc495610(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3549, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f02a31307015cc4469b39c22ac51fa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3fc74df3abfedf16441aca8fc495610
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fe6f4f2695ad16c7c77e69861404b436(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 16, 1, 49, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 16, 16, 49, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f806701357b7f72ee71db5f39236ad4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6f4f2695ad16c7c77e69861404b436
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aade34522bd8e886a16859e50a64fe65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 288, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 288, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a9e9a7621f4f62d33e50ee77c026846f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aade34522bd8e886a16859e50a64fe65
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da79dd60741bbd1567e79af19c32f84c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da79dd60741bbd1567e79af19c32f84c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da79dd60741bbd1567e79af19c32f84c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da79dd60741bbd1567e79af19c32f84c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20d3f909c693bc3384a78029f743e264(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365076adc3d102e71c719de9ef08dd8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 136, 136], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_331c766b1e46eba68efe7224ea68276a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42e5cbf8636ab2328cb24d25e02d55a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25055b0954dbb92ce33c2b7b5fe8843c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dba7f635df5585d7dd15cf6f26aa306e
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_168e7c37c39aba44abc29c2bce01caf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.39313334226608276], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6feb0d50ecf7b32d85ee0a4319cbf055(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed9f40856a970d54ffe24723e19107
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_50a5afcc62a7782a74facdf157325275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39be55716b940d949c945071172b4740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35dd98d8471b30324598d613660a778f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.1350605487823486, 1.9385416507720947, 1.984273910522461, 1.9403074979782104], dtype='float32').reshape([4]),
            paddle.to_tensor([0.959104597568512, 0.902582585811615, 0.7177987098693848, 0.6483383774757385], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_2f039daabfa7ce1ca78b60bd54694b36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.123112678527832, 2.2015790939331055, 2.2268190383911133, 2.111116886138916], dtype='float32').reshape([4]),
            paddle.to_tensor([0.04089539870619774, 0.0974174216389656, 0.28220126032829285, 0.3516616225242615], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_b4c0275d2f8b94eda9963b630693e5b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5336430072784424, 0.49104154109954834, 0.5131800770759583, 0.5000936388969421], dtype='float32').reshape([4]),
            paddle.to_tensor([0.05217211693525314, 0.11233081668615341, 0.15461765229701996, 0.09835059940814972], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_7bdfc8d95d5881032d537aaa3b422cec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7bdfc8d95d5881032d537aaa3b422cec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7bdfc8d95d5881032d537aaa3b422cec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7bdfc8d95d5881032d537aaa3b422cec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f7be09b684bb609a33e2c4e00303220(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7308ce4f71bc6b95b05f00e639e2fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b582d948627475bfb33bff57a0c8e698(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14f267a57c56fba4b10b4b448d8d5622(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33e669bf74ae8c6199fad52a6cd08283
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5ac41b03c5c8d6f2781dd1dfd095e53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f65c315ee5b1f776a3c66ea436faaeb
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e45eaebc956fdf8d427bedc5cdaef6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8570defc04449c14d5bee6ebca4fed58
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3321cbfbdd1af6ad1050273abae0e404(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db95b78c01a0cd6760b50ec1358a26e7
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9a423e03901bccbf4130859edcc903a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_517132dc8f42818cbc7bdb08856c1b3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a423e03901bccbf4130859edcc903a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22626560926437378], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b073031ef0b6591773e92427bc7d210f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_af69164abb41fb29246c3b132a02b0e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.11507900059223175]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.020793229341506958]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_82bf777ea395e5bddedbcb1f24e9502a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09831199049949646]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.16662421822547913]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_0bf8b5a300f147e3a2c0fc19e6f000dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11601881682872772]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.033706799149513245]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_f806701357b7f72ee71db5f39236ad4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6f4f2695ad16c7c77e69861404b436
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d90cd73bf33d8cc3fb5b71522b02de6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.028997577726840973]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.09968048334121704], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_66951fa44ecba49915fe84e243d2c14b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.03644874691963196], [-0.24737341701984406], [0.2195451557636261], [-0.006840735673904419], [0.40923231840133667], [0.04443297162652016]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.22711677849292755], [-0.12359800934791565], [0.42259618639945984], [-0.03915456682443619], [-0.09434542059898376], [-0.01764783263206482]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_497dbad5eabbacf9b7acf1c7b2594d10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0508856326341629], [-0.2884923815727234], [-0.21857397258281708], [-0.11254798620939255], [-0.024763822555541992], [0.416293740272522]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.3462250232696533], [0.081644207239151], [0.09968048334121704], [0.20811206102371216], [0.029323697090148926], [-0.21309828758239746]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_865c4c53d1f5041a7714b4917b7f19fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07023899257183075], [-0.23578545451164246], [0.2195451557636261], [0.09560714662075043], [0.40923231840133667], [0.43172913789749146]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.21945008635520935], [0.1829712837934494], [0.42259618639945984], [0.4049801826477051], [0.029323697090148926], [-0.01764783263206482]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_5d9e5165720e32370b69422269702b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8ce5b019ecea14eeab1e8821734641b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eef93fb848df67dc44565fa01d2b6aa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed9f40856a970d54ffe24723e19107
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0bce3971251283f2cba69aad209b8914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_077b080259d38ef4229744f4eb9ffabd
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d79c954f6db73c3c92e1207fa1f33e96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_379f999eb4bce558d918c04cb94da334
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_756fa99dd4e4e82173538c8d3fa132ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365076adc3d102e71c719de9ef08dd8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90f505cfd6b30c97be35a2942924bb29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 184], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef38159a9bcd4f75467fe700efca1767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93f34d002b99b46d05d1eb89caaa868d
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.6904653310775757]], [[0.8374135494232178]], [[0.6890628933906555]], [[0.7046430110931396]], [[0.7598949074745178]], [[0.7333047986030579]], [[0.6614923477172852]], [[0.8202497363090515]], [[0.7207697629928589]], [[0.765110433101654]], [[0.8303321003913879]], [[0.7572258710861206]], [[0.7418248653411865]], [[0.670176088809967]], [[0.6636852622032166]], [[0.6076679825782776]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_be0f2c6e49f62d20309b1243f37e023f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c5233add8cdd8c5d32d11ce72b27a52
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1abc54e376fb5052102022a91b98959e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39be55716b940d949c945071172b4740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_958b4118d1b39a976a5d1e7cad4591a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de41dd47297a46f2044aa6bf6d6b5a35
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ad7eaf9eea797e6652fe035dae0d9ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f35fb909862ead27f3bdb6714839bc40
    def get_inputs(self):
        return [
            paddle.uniform([145, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_425ede4e3b81ae015020e94c35bba907(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0cd5fa6582c6fa47ad19ded5110ad5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.48927974700927734], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_868bbc1015b501a18caf5be2a7bcc4de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5de0e4ce8fa8364a392f3166e466ad4
    def get_inputs(self):
        return [
            paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_21ef7f363354580c538accf3d7c28886(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f3b25627e85604508691e1d41becd5e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21ef7f363354580c538accf3d7c28886
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5514196fe16e168c00cdf88b5d013cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5514196fe16e168c00cdf88b5d013cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15e9618881bbcf2704b6c3e42456f235
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01fca508ff4a41698ce9963e729a14da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98420ee273a186f34c945b66bbab6eef
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24462288618087769]]], dtype='float32').reshape([1, 1, 1]),
        ]


class PrimitiveOp_74b91555817feac39979146252c7a32a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4116, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c5d6aa179994866dd6fcabb89f232db6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74b91555817feac39979146252c7a32a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e227a7d95b9c12a590ba1b5a044acc0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f65c315ee5b1f776a3c66ea436faaeb
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3b4ab9361655d2fe5948b0646c1c24a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 400, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 400, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_21ab6f4e75e2c5b1b28e5e50df2110db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b4ab9361655d2fe5948b0646c1c24a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_944e83c51e8d121647a84e6e92a6768b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1b8001c62c0bd69cf5de364d5b27a34
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_81d71b176c2019416a00fda3ed0a8582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39be55716b940d949c945071172b4740
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_62c18f93eea105ac2098b9e57dd70260(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39be55716b940d949c945071172b4740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f0050d3cbc1aef0b73f7bc36c6e49a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f0050d3cbc1aef0b73f7bc36c6e49a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f0050d3cbc1aef0b73f7bc36c6e49a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1f0050d3cbc1aef0b73f7bc36c6e49a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_63092bdd0f95c10e3dca7fabd98297bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a765bc0a04d60f18d97a972284ed0241
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_251492e55a3314ec015f4d42e8052794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8ce5b019ecea14eeab1e8821734641b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4ea38b9433fe5130b2855adf0c9df93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c5233add8cdd8c5d32d11ce72b27a52
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e71cb2b5086e992dc023c4ab13add7b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f65c315ee5b1f776a3c66ea436faaeb
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eac66fad68674381fa58147cf3206717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8570defc04449c14d5bee6ebca4fed58
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7b5aa0775d0163db95da8cd81dd59d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6558a47673440af5766878b002e3a19c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_697f5999308dfb7f1532283236f37c92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 336, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 336, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1b924fdec3cc23ba72b287ad7b40d688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_697f5999308dfb7f1532283236f37c92
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66af5a2abeb2ad3f0e348b66e7a7326d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a8ad0cdc472c07c09f80ce9a67cb05c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f65c315ee5b1f776a3c66ea436faaeb
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13efc424bbb13915b3fece9609885532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91aa37acc2c39060693c7c183cd1c99d
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36f01420ed2327f7cf7f3850d440b720(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64b3661e08a9412b02da4daf9ca243b1
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14a4f43aee5569af215eff3d34af6794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_077b080259d38ef4229744f4eb9ffabd
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73872a41564e7527a5641dec82098953(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365076adc3d102e71c719de9ef08dd8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_289da9e370b87b8230fefc86aef88388(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_289da9e370b87b8230fefc86aef88388(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_289da9e370b87b8230fefc86aef88388(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a9696e30909f6bb43629b7b84126fa0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6558a47673440af5766878b002e3a19c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_16d68d6cd9a76b72250d12c789069b0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a69e43a060f8a1b0a2aac3060ad88535
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_163bd5955706b238fc93fa07c1e35b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8ce5b019ecea14eeab1e8821734641b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_51587ad9decc41aae77ee442776d2b85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b4ab9361655d2fe5948b0646c1c24a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ef98ea311740ef38bd6c5910f9f8b0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14304721355438232], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e565ee2acf9ee57916d7a11375fba47c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ba25f242d8af46e2756a95fffc150d0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c2bb25f28973667d3abb75ac6decf547(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba25f242d8af46e2756a95fffc150d0b
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_29d6a6bc7bef6885014c49f2880fdd30(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 56, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 56, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_08a438cd6c2f77b77aab6ec6eff612da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29d6a6bc7bef6885014c49f2880fdd30
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab7b1893b50be5a1f2a33e7b95ddccaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f078b579ac35dac495d2ec451fc4f7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f078b579ac35dac495d2ec451fc4f7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f078b579ac35dac495d2ec451fc4f7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f078b579ac35dac495d2ec451fc4f7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f078b579ac35dac495d2ec451fc4f7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4df63079adabbc0e2cd32edc4803a2f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4df63079adabbc0e2cd32edc4803a2f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f078b579ac35dac495d2ec451fc4f7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89038fee494c05b5c4b527f45ddf9fe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42e5cbf8636ab2328cb24d25e02d55a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60169953cbfa8d88d6fd18c3171c4095(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8612514138221741], dtype='float32').reshape([1]),
            paddle.to_tensor([0.14591339230537415], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b7a50208f09ce0de262e6d91ed74453d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7785527110099792], dtype='float32').reshape([1]),
            paddle.to_tensor([0.2627646327018738], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f7cb33100ec9e48c88475c36e6491cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.6168838739395142], dtype='float32').reshape([1]),
            paddle.to_tensor([0.21462267637252808], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_85165957574d8223b3c95a22e2b51da0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.713913083076477], dtype='float32').reshape([1]),
            paddle.to_tensor([0.21841007471084595], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f9cd22ac82803a9a8286dea49ad2cdd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.9243948459625244], dtype='float32').reshape([1]),
            paddle.to_tensor([0.37511470913887024], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_91ad49d617080e7526e582a003844bc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8532863855361938], dtype='float32').reshape([1]),
            paddle.to_tensor([0.15417197346687317], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2886dc40d8e9ae9d8732c3007b04e30a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.6298309564590454], dtype='float32').reshape([1]),
            paddle.to_tensor([0.061254553496837616], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_34c144d82a51cd86d59716cda2cc4e14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8147094249725342], dtype='float32').reshape([1]),
            paddle.to_tensor([0.4080032706260681], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d477be711cf689475f6063734402c7a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.991642415523529], dtype='float32').reshape([1]),
            paddle.to_tensor([0.2252350151538849], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_17b94d78918f3477084704ef69e1303a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f65c315ee5b1f776a3c66ea436faaeb
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2418ad7e7ede2ae716ce43afa2d876f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2418ad7e7ede2ae716ce43afa2d876f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2418ad7e7ede2ae716ce43afa2d876f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2418ad7e7ede2ae716ce43afa2d876f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d66501acc8efb60a1eb4900db994d46e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4480fb5fe8259dd5c165094e2ca2941d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8ce5b019ecea14eeab1e8821734641b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 42, 42], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a4881148a414d4ac6275d85c3f510722(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_077b080259d38ef4229744f4eb9ffabd
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_caf251ae6dc5e7e6ea88552a44b112a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b679c4cc71d59f2e43b2dfcd94557384
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.08972160518169403], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_caf251ae6dc5e7e6ea88552a44b112a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b679c4cc71d59f2e43b2dfcd94557384
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.08972160518169403], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2b8a78a3abc7c4452be179435e14f690(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fe80bf45124ca8664ea8278252b5644
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_16da1765b91c5d4704dadf8d5bf7ed49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e729ce4a58e1fb04f064fdd49605571c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21743687987327576], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9f4dff7c7c30b5d8315ce1477dd3562b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7308ce4f71bc6b95b05f00e639e2fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9230c263ee9f4613278df33412e1506(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6558a47673440af5766878b002e3a19c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_359bb455b9291f94ba1cafc825a49d16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_359bb455b9291f94ba1cafc825a49d16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_359bb455b9291f94ba1cafc825a49d16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_359bb455b9291f94ba1cafc825a49d16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_359bb455b9291f94ba1cafc825a49d16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0374102387e250b638a54b89c6d41d19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([4618, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0374102387e250b638a54b89c6d41d19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([4618, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_359bb455b9291f94ba1cafc825a49d16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4618, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_700f00bcfbbb1f4307af62e00093f8c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_077b080259d38ef4229744f4eb9ffabd
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e63b527e1fd25c01c06029d1b7a69602(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_233c3eb52c21b71f0cf7410d542ffde6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e63b527e1fd25c01c06029d1b7a69602
    def get_inputs(self):
        return [
            paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.37241241335868835]], [[0.2434406727552414]], [[0.17202362418174744]], [[0.4024975895881653]], [[0.46557819843292236]], [[0.3065173923969269]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_f345436a6de7eb6761742ce621005ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42e5cbf8636ab2328cb24d25e02d55a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4214cb42d6f1056aa2cefe8a478c694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4214cb42d6f1056aa2cefe8a478c694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4214cb42d6f1056aa2cefe8a478c694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4214cb42d6f1056aa2cefe8a478c694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4214cb42d6f1056aa2cefe8a478c694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9077afe9eb4789de4d57680e699f5a3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1058, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9077afe9eb4789de4d57680e699f5a3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1058, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4214cb42d6f1056aa2cefe8a478c694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3bfb480cf9f60349cffdb11e4b40900(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24220e97cfb882d7db6688d0b0dceb7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4c0acd54ee1932d93183c0253341a79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b38516e087e509e2fd8e3adcbacad0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14b7d19fc9c49dd52d4f2e924a6b5fd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b695408f744c53c4e498b8eca10c146(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3430aac1e87ac914d423761d92512b31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ae2f2e513038cea3b709c5fa18bdc2a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3430aac1e87ac914d423761d92512b31
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cd059d2c0aca693a6d0bd85fcfc31034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3430aac1e87ac914d423761d92512b31
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_999f7ce0c801a7e6299f55ff81a2d5f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3430aac1e87ac914d423761d92512b31
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4425f2e6a84dd371ea7dbee04852bb50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3430aac1e87ac914d423761d92512b31
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_019a428061f24ae4abcedf66d37b7dd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3e4323a567951a3d707091b7981614b
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa014bc9476b17053059c8fe747ed9d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8ce5b019ecea14eeab1e8821734641b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0bce3971251283f2cba69aad209b8914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_077b080259d38ef4229744f4eb9ffabd
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25baf12596500cb4f8f3340e362df1bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c5233add8cdd8c5d32d11ce72b27a52
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_39d6def76ee1fd3e178286b0e8b54484(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a27c74de460ece435090f3e7f32c4d9e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ef45dcd3ae0df98a8bfad37f110b9f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ef45dcd3ae0df98a8bfad37f110b9f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ef45dcd3ae0df98a8bfad37f110b9f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d9c5585d219fa5fb0f1f420a01927a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a27c74de460ece435090f3e7f32c4d9e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_adcce674fc5ad8c69eb49dc1f9b65cb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_adcce674fc5ad8c69eb49dc1f9b65cb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a437d26651388809f8f0a46eec0896a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3710291385650635], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f1d11cbb21763f46099e7ce511efb795(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a27c74de460ece435090f3e7f32c4d9e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bf3088cc16f31c965831062ca57f3f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bf3088cc16f31c965831062ca57f3f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bf3088cc16f31c965831062ca57f3f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bf3088cc16f31c965831062ca57f3f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d1b97dc50da98911c4d5bcb820acad04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 200, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 200, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b5d031fcd589b38e7b67e67a23583860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1b97dc50da98911c4d5bcb820acad04
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_177590cd52bd62f378486762c1ef9945(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39be55716b940d949c945071172b4740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d5b8478af74dca2c2504ea117ed35028(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 32, 1, 49, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 32, 16, 49, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e168a8b1f854d5d54e68326b2e603957(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5b8478af74dca2c2504ea117ed35028
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ae11320503813de712beff0767170a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_084821d92360c14e4e4d0a29cf72c48f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ae11320503813de712beff0767170a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_084821d92360c14e4e4d0a29cf72c48f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ae11320503813de712beff0767170a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_084821d92360c14e4e4d0a29cf72c48f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_34b1d61fa37d839c9c89ae37750ce176(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2048, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f39a18cee1d0206e4fbade16cab68461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b1d61fa37d839c9c89ae37750ce176
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_442def56b1745511a29bb0ea1952b29c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42e5cbf8636ab2328cb24d25e02d55a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_518dacedf00c3e090ba7a1241a03047b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_518dacedf00c3e090ba7a1241a03047b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c469cd288af24d97812135e92db4fbae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0], [0.010165899991989136], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_7577603a434571fb58e307a7bb2a8dcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.23123082518577576], [0.10221844911575317], [-0.4777134656906128], [-0.21933674812316895], [-0.0261429101228714]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.34980669617652893], [0.010165899991989136], [0.0596519410610199], [-0.07478496432304382], [-0.22150114178657532]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_6db9f57e9758efd789fa3352dec48317(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.46136152744293213], [0.04899318516254425], [-0.12858983874320984], [0.4731298089027405], [-0.16646058857440948]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.14638830721378326], [0.061031460762023926], [-0.253073126077652], [0.20038901269435883], [0.19977667927742004]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_bba5604ef2c6c291294cea7559f7ea4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.23123082518577576], [0.18378254771232605], [-0.12858983874320984], [0.4731298089027405], [0.1682271659374237]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.34980669617652893], [0.061031460762023926], [0.0596519410610199], [0.32782143354415894], [0.19977667927742004]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_ecdc5a397894adb1604738d8e3fb4295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a27c74de460ece435090f3e7f32c4d9e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1371c48d2f29cecf2a3113d306b4eadd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365076adc3d102e71c719de9ef08dd8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad8c94a6920e604b78c14f6a52a6b8da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365076adc3d102e71c719de9ef08dd8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a7839bc29fa408f2645a06ed8d01664(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8570defc04449c14d5bee6ebca4fed58
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d35808d05c7e718499cbd4315f63d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39be55716b940d949c945071172b4740
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7421ac25178a09eef8ef7ed703227ee3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bab70f58b8b7ac4324068002c1bedc50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7421ac25178a09eef8ef7ed703227ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01421f7dff29dbb58036232da4159fa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_3605aca3e677045f208df2206e91238b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_0aaf2290134d34f1cb502ed2c21714d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_879278357d980c22197d2a1a86b4e6d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_6fb1622b9c699d65e2ff5f2bca5df7e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(8, dtype='int32').reshape([]),
            paddle.to_tensor(8, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_4a0aa350ef32d32ced194c3fefdee167(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0cd5fa6582c6fa47ad19ded5110ad5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 76, 116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.28403231501579285], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ab953a1ae6cf3c05223cc92e648000ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f3bc8fd07c5dacb5e606a94f7719f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a618b21f875fc9539d5be7c532f6ef83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39be55716b940d949c945071172b4740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc18ce2544fc573153d3dcbdf639a096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aade34522bd8e886a16859e50a64fe65
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_333dc3c20dcc6619754caa85bd2f1f1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42e5cbf8636ab2328cb24d25e02d55a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ca61d2b81a889ce6d72b8b2a55f104d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fad09e57f2687163480f6dd7e30c1b7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4aa7b666b7d0acf3d7ea88511e816560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a423e03901bccbf4130859edcc903a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4611557722091675], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af56aeeb84822e85d1a96a3c285361c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_084821d92360c14e4e4d0a29cf72c48f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af56aeeb84822e85d1a96a3c285361c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_084821d92360c14e4e4d0a29cf72c48f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af56aeeb84822e85d1a96a3c285361c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_084821d92360c14e4e4d0a29cf72c48f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d24d3b3868593ff8ecc222e0f5db94b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b1d61fa37d839c9c89ae37750ce176
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6bd3d31341a6a3cd46e149d1cde4ba79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1248, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d8c64e8e4dddcfa36307dbc91556240f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bd3d31341a6a3cd46e149d1cde4ba79
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1248, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9bc9ec0e5728384e76d8239644f6a275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aade34522bd8e886a16859e50a64fe65
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5de8bbfc030623c7499852d65806ff9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b679c4cc71d59f2e43b2dfcd94557384
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.23867575824260712], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5de8bbfc030623c7499852d65806ff9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b679c4cc71d59f2e43b2dfcd94557384
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.23867575824260712], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0fb5684e1768a44747951f83105ec7f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fe80bf45124ca8664ea8278252b5644
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db08202f61115f32e66b4166f1356ba8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e729ce4a58e1fb04f064fdd49605571c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15143291652202606], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5868663be13fae4fb58dbb4084b94ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5868663be13fae4fb58dbb4084b94ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5868663be13fae4fb58dbb4084b94ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14d5ba4a9c30007c5cf7cabdee3666f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aed5bda16e8cfde36cccb444f43454e4
    def get_inputs(self):
        return [
            paddle.uniform([171, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4a2605b3435a2b4b649f4a5cfc03b38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4a2605b3435a2b4b649f4a5cfc03b38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4a2605b3435a2b4b649f4a5cfc03b38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_011b4e42f34d6960d873d5e7a8f5d293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_011b4e42f34d6960d873d5e7a8f5d293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_011b4e42f34d6960d873d5e7a8f5d293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_898a310661f073ae0d8c7e769d158b1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db95b78c01a0cd6760b50ec1358a26e7
    def get_inputs(self):
        return [
            paddle.uniform([145, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a2300b5a1fcc84ed9f40a84a3e0ec96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7421ac25178a09eef8ef7ed703227ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d6e52ec69c40f2a7aaa73ef77962b9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.49205729365348816], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9be9050b89f71bd2e085ada6cf708a04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365076adc3d102e71c719de9ef08dd8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c12767ec75bd00b55dc5ddbfa495b86d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed9f40856a970d54ffe24723e19107
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_993590a34fad3b3a59be4ab2eecbaa9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fad09e57f2687163480f6dd7e30c1b7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 23, 23], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a02bf3cf35f111dd7e31b68ba6588abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a02bf3cf35f111dd7e31b68ba6588abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a02bf3cf35f111dd7e31b68ba6588abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a02bf3cf35f111dd7e31b68ba6588abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a02bf3cf35f111dd7e31b68ba6588abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f58720d588402bf2bbc46b5005282e0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([2402, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f58720d588402bf2bbc46b5005282e0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([2402, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a02bf3cf35f111dd7e31b68ba6588abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2402, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_743ac389e85aff4b57e83bad028f33f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c5233add8cdd8c5d32d11ce72b27a52
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b594f9265530bab0ef8c4890f561f69e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39be55716b940d949c945071172b4740
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a69d7eb1b84c59db5ef566f48e6bb48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a69d7eb1b84c59db5ef566f48e6bb48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a69d7eb1b84c59db5ef566f48e6bb48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a69d7eb1b84c59db5ef566f48e6bb48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a69d7eb1b84c59db5ef566f48e6bb48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ede174095826e2ff2a7837c779e724c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ede174095826e2ff2a7837c779e724c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a69d7eb1b84c59db5ef566f48e6bb48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82d8a2dbf86b3219224a177f0ef5b052(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82d8a2dbf86b3219224a177f0ef5b052(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82d8a2dbf86b3219224a177f0ef5b052(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82d8a2dbf86b3219224a177f0ef5b052(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82d8a2dbf86b3219224a177f0ef5b052(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a8b0aff05c3e346270d4c5c76052c3cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a8b0aff05c3e346270d4c5c76052c3cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82d8a2dbf86b3219224a177f0ef5b052(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4d2dc5fc1c04328db7f2da91052211e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b679c4cc71d59f2e43b2dfcd94557384
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.23298466205596924], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c4d2dc5fc1c04328db7f2da91052211e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b679c4cc71d59f2e43b2dfcd94557384
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.23298466205596924], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3ac8245bfd4c58bc609b7e89307a0a92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fe80bf45124ca8664ea8278252b5644
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_629944fce3320b7a22d0209dfe7d572f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e729ce4a58e1fb04f064fdd49605571c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.30240482091903687], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_bd53eaf54790773621796d3b2a1819df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 156, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_04615ccd1b6539e8756d079a46ef113b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd53eaf54790773621796d3b2a1819df
    def get_inputs(self):
        return [
            paddle.uniform([1, 156, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 156, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_06afa0ac8681719902f3e86d0feb1b9c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_82ca53b3816d340384da9fa4983d3c2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06afa0ac8681719902f3e86d0feb1b9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.8765795826911926]], [[0.9087433218955994]], [[0.9128962159156799]], [[0.8527706861495972]], [[0.8580156564712524]], [[0.8770037293434143]], [[0.8779484629631042]], [[0.90641188621521]], [[0.8966494798660278]], [[0.9234762787818909]], [[0.9084265828132629]], [[0.8563979268074036]], [[0.9175929427146912]], [[0.9030978679656982]], [[0.9414277672767639]], [[0.9114814400672913]], [[0.9379655122756958]], [[0.8534589409828186]], [[0.9528156518936157]], [[0.9564563632011414]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class PrimitiveOp_08a35fac70f021394b01b6711933f04f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 40, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20a228bd881e6abe9c2150c1edcb899f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08a35fac70f021394b01b6711933f04f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_34967eb0bb98058b9f8a9a8cc0d2f7e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 32, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b98eb3e500bcacfd0b31f99ca2396226(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34967eb0bb98058b9f8a9a8cc0d2f7e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5ae2202f46a54d568740a3992dd4d5a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 160, 16, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e90e446b4d6a7053fda61404189f3e64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ae2202f46a54d568740a3992dd4d5a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_054f5187721a79ef1c529b16e3902ac6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c5233add8cdd8c5d32d11ce72b27a52
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 92, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69a58e50ba723730f651fa8a74868a8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8d51264be6beb2c37dedec2f3a743110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c5233add8cdd8c5d32d11ce72b27a52
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc3aa6ff9951515bc0a5860ffce3afb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1b97dc50da98911c4d5bcb820acad04
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e7505f7bc157b6854dfd2cc4a6cc80b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4147b54f8482f85d2da05c4d03cbfe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_379f999eb4bce558d918c04cb94da334
    def get_inputs(self):
        return [
            paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_8eabaa3269588cc668e7b36edf75b853(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cf52968ff726dfa7ab970ab9887cee8
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2418ad7e7ede2ae716ce43afa2d876f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2418ad7e7ede2ae716ce43afa2d876f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2418ad7e7ede2ae716ce43afa2d876f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d08d634b4e6a20875557271d441572be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a423e03901bccbf4130859edcc903a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.31990066170692444], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9513613484b790b436f9e8d705cb9da8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e80edc4e2a448be1178373c64e8d2f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_510ddc2ea4dbc588738205f3a468fdbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8ce5b019ecea14eeab1e8821734641b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21f330f1b74e5cd291a7e46e35522bc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8570defc04449c14d5bee6ebca4fed58
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa2cc991001f630c23656cdb2bd1dcb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42e5cbf8636ab2328cb24d25e02d55a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c43d32e698ed2ae35037971f3a1c58e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1ca419767c56c49909af1ef11eb453c
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a50aa2dd342514e148d910d7c7377100(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d212094f49c51f92ec8037edd298eeea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fa34e62f4f2ceafc1c2d9505b9015908(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 1, 49, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 8, 16, 49, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e459d2bd856669f2dea44957f175396(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa34e62f4f2ceafc1c2d9505b9015908
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_68b1296fd7c36b8e58fefde3b48325d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aed5bda16e8cfde36cccb444f43454e4
    def get_inputs(self):
        return [
            paddle.uniform([22, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4da612ebf1274aa50522cdba5e320a0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9984ef09cd593fb4560d8d5176e4c7bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a2473bad63f20d04b32490d59023673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aed5bda16e8cfde36cccb444f43454e4
    def get_inputs(self):
        return [
            paddle.uniform([145, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_25eafde30d2b600ec4757221ece16ec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.36362195014953613], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_dc92fca6b0d576aa7e787ec52e99bc7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_251c5113f43dc4319e32962d11836656(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc92fca6b0d576aa7e787ec52e99bc7c
    def get_inputs(self):
        return [
            paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.28761032223701477]], [[0.322146475315094]]], dtype='float32').reshape([2, 1, 1]),
        ]


class TestPrimitiveOp_598ba36488adda70901bdfe72f2a4a6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42e5cbf8636ab2328cb24d25e02d55a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 46, 46], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a110d518a795d074b2a61e32baffa205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db95b78c01a0cd6760b50ec1358a26e7
    def get_inputs(self):
        return [
            paddle.uniform([171, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ad4d8b2af8f1e48e6c666ef31b8c2059(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 1, 49, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 16, 16, 49, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70746e0deb8d731f2134be47100661ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad4d8b2af8f1e48e6c666ef31b8c2059
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c27f247f5e9670c1ca8b2cb8a466695a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8ce5b019ecea14eeab1e8821734641b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_00b5c3adf4c2df89caf66042439ff1b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c72b5a65081e1bea3a6109e268864e27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00b5c3adf4c2df89caf66042439ff1b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ff87fb7093bf46933af4ec67f2ddc0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed9f40856a970d54ffe24723e19107
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9fdee5cb855466bd8f8c60610748b510(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc575bcb7876f61b7fe355891febd35b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fdee5cb855466bd8f8c60610748b510
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5e7328134afc4778853ce549f039d523(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aade34522bd8e886a16859e50a64fe65
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_790f78bb904d1b4385831401401b9ae3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_391ea07d377bd44a525315beafa2d798
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5fa5348790b760007068d5db1f909bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68c48677c7bbaf3489b1be408a6d1c47
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0581e35af339f2f2d2a529bfe2469af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.0393970012664795, 1.9818875789642334, 2.087188959121704, 1.873114824295044, 2.1052823066711426, 1.864662766456604, 2.2452821731567383, 1.970820426940918, 2.2432594299316406, 2.2626304626464844, 1.9228367805480957, 1.973318338394165, 2.072862148284912, 2.3036062717437744, 2.054995059967041, 1.96905517578125, 1.9071295261383057, 2.0975520610809326, 2.1245319843292236, 2.1823770999908447], dtype='float32').reshape([20]),
            paddle.to_tensor([0.7149816751480103, 0.5398354530334473, 0.5755943059921265, 0.930835485458374, 0.9943198561668396, 0.7548472881317139, 0.9924588203430176, 0.6865043044090271, 0.6105813980102539, 0.7195262908935547, 0.5569946765899658, 0.7612473964691162, 0.9669315814971924, 0.7985126972198486, 0.9640113115310669, 0.955947995185852, 0.7569136619567871, 0.9921732544898987, 0.773321270942688, 0.6479578018188477], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_b63721b1275ae434831ed8210f8c2e8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.2356059551239014, 2.1325418949127197, 1.9988677501678467, 1.9358450174331665, 2.070831775665283, 2.1218035221099854, 2.1543352603912354, 1.9613558053970337, 2.1132144927978516, 1.915096640586853, 2.14644455909729, 2.1328470706939697, 2.025054931640625, 1.9914488792419434, 1.8980238437652588, 1.918556809425354, 2.2420969009399414, 2.1258935928344727, 2.0423648357391357, 1.969955563545227], dtype='float32').reshape([20]),
            paddle.to_tensor([0.28501829504966736, 0.46016451716423035, 0.42440566420555115, 0.06916454434394836, 0.00568016804754734, 0.24515274167060852, 0.0075411563739180565, 0.3134956955909729, 0.3894185721874237, 0.2804737091064453, 0.44300535321235657, 0.23875261843204498, 0.03306839242577553, 0.20148731768131256, 0.03598868101835251, 0.04405199736356735, 0.2430863231420517, 0.007826716639101505, 0.22667871415615082, 0.35204222798347473], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_f039c66e75f9118dda79c311f8fa0202(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.523829996585846, 0.512803316116333, 0.5124261975288391, 0.46936339139938354, 0.526271641254425, 0.48192542791366577, 0.5611490607261658, 0.49196332693099976, 0.548154354095459, 0.5412890911102295, 0.5054740905761719, 0.5028515458106995, 0.5178202986717224, 0.5601776242256165, 0.5123364329338074, 0.4917076528072357, 0.4971388578414917, 0.5244434475898743, 0.5264766216278076, 0.526898980140686], dtype='float32').reshape([20]),
            paddle.to_tensor([0.2586003243923187, 0.4573405683040619, 0.45736679434776306, 0.09587918967008591, 0.02150222286581993, 0.0755179151892662, 0.02047641947865486, 0.2755345404148102, 0.488018661737442, 0.16947628557682037, 0.27590978145599365, 0.2902534604072571, 0.289183646440506, 0.30879873037338257, 0.40384769439697266, 0.3183794319629669, 0.4515976011753082, 0.1335342675447464, 0.42282000184059143, 0.3426266014575958], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_4291408d086d3a5ec242c08ef8a6b92c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8570defc04449c14d5bee6ebca4fed58
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f551af64b8ec582461adfc573a8438dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fed3fe86ac708f648c6763ee11663a84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f551af64b8ec582461adfc573a8438dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20f61356260405197b9294080efaa687(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8570defc04449c14d5bee6ebca4fed58
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e2d6431001e8d4723cbbfd81bea9a0d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5de0e4ce8fa8364a392f3166e466ad4
    def get_inputs(self):
        return [
            paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01576eed5c9e85151588310ae0502b74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21ef7f363354580c538accf3d7c28886
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d3ff13d07277be506a9df9eab09623b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39be55716b940d949c945071172b4740
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 84, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ac475292e86b18f7ebdf524b22e81c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a27c74de460ece435090f3e7f32c4d9e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5d1e84a54a60ea75239b55f3b84217f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b73d20cebb10645d031e52e745d0cf6
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.933100700378418]], [[0.9223877191543579]], [[0.9466491937637329]], [[0.801755964756012]], [[0.9523014426231384]], [[0.963595986366272]], [[0.9067689776420593]], [[0.8882817029953003]], [[0.9549538493156433]], [[0.9347421526908875]], [[0.8766946196556091]], [[0.8824866414070129]], [[0.8885610103607178]], [[0.863728404045105]], [[0.9099065065383911]], [[0.9257481694221497]], [[0.9844434261322021]], [[0.9343804717063904]], [[0.9427490234375]], [[0.9508345723152161]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_6770c548d1849d859d5eec9a24f05381(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9984ef09cd593fb4560d8d5176e4c7bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ff48a8f92c5de664bedcf6c697b4042(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6558a47673440af5766878b002e3a19c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e80edc4e2a448be1178373c64e8d2f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0aaf2290134d34f1cb502ed2c21714d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_3605aca3e677045f208df2206e91238b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_01421f7dff29dbb58036232da4159fa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_9ae0b7422085c51fa002b7dca3e1abb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7308ce4f71bc6b95b05f00e639e2fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28119637b5e500ed0719d28e2a79da29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f65c315ee5b1f776a3c66ea436faaeb
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21f330f1b74e5cd291a7e46e35522bc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8570defc04449c14d5bee6ebca4fed58
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8a298ba96af8ed84bcda57da776c37aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365076adc3d102e71c719de9ef08dd8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14a4f43aee5569af215eff3d34af6794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_077b080259d38ef4229744f4eb9ffabd
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_661479d087e40eca06b5846c5dea92c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33e669bf74ae8c6199fad52a6cd08283
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cd3064444e6f091f93fc42ba2c0ad5e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_495d91e33a635b19edea897af6cd1867
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8e459d2bd856669f2dea44957f175396(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa34e62f4f2ceafc1c2d9505b9015908
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_80a2642d2c0aa3deb3160231c27c4d90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_077b080259d38ef4229744f4eb9ffabd
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04776d8469653b881efca104140ec295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_697f5999308dfb7f1532283236f37c92
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0793232adcf60b5f2e3a32d449f3764a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_077b080259d38ef4229744f4eb9ffabd
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_adb474579bc42725676d9c14277e32e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42e5cbf8636ab2328cb24d25e02d55a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_81f9f7874426f0ebee93c63b5e805fd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29d6a6bc7bef6885014c49f2880fdd30
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4bd8cc45fca1f63e0bb9af7d7638616b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5de0e4ce8fa8364a392f3166e466ad4
    def get_inputs(self):
        return [
            paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9fb7d6b38eb9a809f4fd0d37939beae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_273f5fa5e1a60bc6bab9fc18b1756fdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23dbd70359fb5735aaedcd0ec7b94752(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1685902625322342], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.20464426279067993], [0.0]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_b19851e13f6147eeb5c2705ecd6ab5b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2085653394460678], [-0.13071531057357788], [-0.4041455388069153], [-0.22420132160186768]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.18431469798088074], [-0.29548147320747375], [0.20464426279067993], [0.18481111526489258]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_658d07fbbbc0117b843f44b3401de3c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2211240977048874], [-0.04705144464969635], [-0.019578903913497925], [-0.180839404463768]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.018745750188827515], [0.14113877713680267], [0.32845786213874817], [-0.1698584258556366]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_ae3720127eac56d06eb79ce7a926b0c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2610991597175598], [0.14274170994758606], [-0.019578903913497925], [0.05140325427055359]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.018745750188827515], [0.14113877713680267], [0.32845786213874817], [0.18481111526489258]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_465fd319de580febedcfaa58e5db726f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6558a47673440af5766878b002e3a19c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6c318da2e143be2f423a053389bf091(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e899c64cf254931f2d3bdac698e40af6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8ce5b019ecea14eeab1e8821734641b
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a5264961f316270cb0f678ddb3f7910e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cf52968ff726dfa7ab970ab9887cee8
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f4d17a49cb0e5ac4d61a570abd83fef8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 576, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_71bbd03bb1c114ab2fd685ecb62c5c50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4d17a49cb0e5ac4d61a570abd83fef8
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dace524da911487b369022e70ee39f28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a27c74de460ece435090f3e7f32c4d9e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57fd8af3c27f180f6ae177784506a3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_379f999eb4bce558d918c04cb94da334
    def get_inputs(self):
        return [
            paddle.uniform([43, 40, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_518898b65acdb90280ac87bf188e9b86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a765bc0a04d60f18d97a972284ed0241
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f71b20c8ffbb59324709da928e840a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c5233add8cdd8c5d32d11ce72b27a52
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c2904ae21c3222442090643347f655b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7308ce4f71bc6b95b05f00e639e2fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67756860e9567a832eaac5ee2bd31a4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67756860e9567a832eaac5ee2bd31a4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67756860e9567a832eaac5ee2bd31a4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67756860e9567a832eaac5ee2bd31a4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67756860e9567a832eaac5ee2bd31a4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07699ddf95248d305994e56bcbbcf6aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([2114, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07699ddf95248d305994e56bcbbcf6aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([2114, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67756860e9567a832eaac5ee2bd31a4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2114, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7923904033b5954ecbb2109b2b33fbd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_077b080259d38ef4229744f4eb9ffabd
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2c37b3f38039b15f986d269493c9ff8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed9f40856a970d54ffe24723e19107
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c64a9b9541e14e07683ff5107e945b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a765bc0a04d60f18d97a972284ed0241
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 21, 21], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_baf4a53208cde2603284bca97cdd2a58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_084821d92360c14e4e4d0a29cf72c48f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eb2f257740f2307f49fe3ccaa89b851b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c5233add8cdd8c5d32d11ce72b27a52
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b1a52ce74cadc18cd583c06438657b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f65c315ee5b1f776a3c66ea436faaeb
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ac334a99cfefff7aa7dea18a98b0d7d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_37f2397a5335f37c20ac4f141031b1ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac334a99cfefff7aa7dea18a98b0d7d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb9c1b5009150def50f5613acdcfd07d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a423e03901bccbf4130859edcc903a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.32011333107948303], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34fcd4a324a23be1923e3e2741ad7f92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a27c74de460ece435090f3e7f32c4d9e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2472b300c494153848ce36201cd5f902(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 2, 1, 9, 112, 112], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 2, 16, 9, 112, 112], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97a04b1b0ffcbd3b20313f75067cc4b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2472b300c494153848ce36201cd5f902
    def get_inputs(self):
        return [
            paddle.uniform([22, 2, 1, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47f460b6eeacaeffdc9c2b95503469ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b679c4cc71d59f2e43b2dfcd94557384
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3379891514778137], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_47f460b6eeacaeffdc9c2b95503469ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b679c4cc71d59f2e43b2dfcd94557384
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3379891514778137], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cbab3b741b6ad3ef14d5d0b1eefec635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fe80bf45124ca8664ea8278252b5644
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eed1de4a17a904c2dff0184e98fa0b45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e729ce4a58e1fb04f064fdd49605571c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.42617350816726685], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4b09630682bc2ded1cecba6ee55d46b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1b97dc50da98911c4d5bcb820acad04
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53145961f15bfbe89b37fb8a7b302759(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b4ab9361655d2fe5948b0646c1c24a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a43aade7b7656a8c4a6f887a32b645b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d409b9b7e56c506f1f394e82ed802f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13efc424bbb13915b3fece9609885532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91aa37acc2c39060693c7c183cd1c99d
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fff741ae09b4d773d99ff30ec7672aa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_011b4e42f34d6960d873d5e7a8f5d293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_011b4e42f34d6960d873d5e7a8f5d293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_011b4e42f34d6960d873d5e7a8f5d293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_011b4e42f34d6960d873d5e7a8f5d293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e168a8b1f854d5d54e68326b2e603957(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5b8478af74dca2c2504ea117ed35028
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_43c98cd1347c40d5b42d87c5905da211(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f3bc8fd07c5dacb5e606a94f7719f18
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe743c2606a69acd1286ac7b4b988e8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e735a95fb5465512540f88729564065a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365076adc3d102e71c719de9ef08dd8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_da259e892bc889d4e17184e842d2110b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 288, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a754653742a1c5fcb9254959d6a8688c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da259e892bc889d4e17184e842d2110b
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7a8d59909ba35e735c5601d021027f1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_29011776bb501c2a7449822400f1dc74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91aa37acc2c39060693c7c183cd1c99d
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b83079c8e510360c3f05b8bea7211ff4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8ce5b019ecea14eeab1e8821734641b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c9a96fec48bf6f8427deec2fbcd2fed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7308ce4f71bc6b95b05f00e639e2fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5642d80cfdb714deb56b91d1083bd8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6cf52968ff726dfa7ab970ab9887cee8
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af56aeeb84822e85d1a96a3c285361c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_084821d92360c14e4e4d0a29cf72c48f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_082b99fab11286358636c3b140538ffd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_495d91e33a635b19edea897af6cd1867
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c6b8a4052a960cd049549db353cb158(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0cd5fa6582c6fa47ad19ded5110ad5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 38, 58], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3613332509994507], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_81d71b176c2019416a00fda3ed0a8582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39be55716b940d949c945071172b4740
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c74785e91eed091c5bd0bff16eeb031(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_609a2913b2fe5dfa3c97e26b3fedb070(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4246678650379181], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2986fa98826ce45c274f2b4072a785c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7308ce4f71bc6b95b05f00e639e2fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e701afcaf09ae0df6cf016ab6f0207e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8ce5b019ecea14eeab1e8821734641b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8da80925aa2163f3ebf62d9bbdff67ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75ab1ce1ca742268bc1923ad626c28f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a8f041a15dd13cf0dea27b5484d51fe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0cd5fa6582c6fa47ad19ded5110ad5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.028038688004016876], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b594f9265530bab0ef8c4890f561f69e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39be55716b940d949c945071172b4740
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e115bb93acf510bf890857f92bffadc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae180e04ef411d9166814961f037b343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae180e04ef411d9166814961f037b343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae180e04ef411d9166814961f037b343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae180e04ef411d9166814961f037b343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_422abe390ccf1a2220e9bb8248ae4490(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_08428af55bf5c44487723fcb2ef68853(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_422abe390ccf1a2220e9bb8248ae4490
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_803bef613a81dcd46357f96a80c49fe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a739ab0473dad64947d4332d09e47e98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c24bc0d8c8572cd6f279f2c0b5e0f41c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7120aabc6b58f0706380ae573accc28b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1401c4b95a96c24befe11ab0c12cde46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3430aac1e87ac914d423761d92512b31
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_d04f5569f0065af40469c8783f265e2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3430aac1e87ac914d423761d92512b31
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_7833e8cd1f935f7b7ab75504368d7749(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3430aac1e87ac914d423761d92512b31
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_77adb4d68db3b0f0093d2eb2726895d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3430aac1e87ac914d423761d92512b31
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_6a306b592ce1db5d4138688ea8aeccca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3056226372718811], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5724536a97510f97d98a3c7a75cbee98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a765bc0a04d60f18d97a972284ed0241
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5aced072f9504fd7bf79505f4df8b214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aade34522bd8e886a16859e50a64fe65
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae3d6dc9230440962a951bcc6cf55409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_077b080259d38ef4229744f4eb9ffabd
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_41995c0e7332aa9f59b64533db54db93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3366011381149292], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_df1201702b34d1a5443f389b4689f96b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeacd4c721523530e52c46b66de64a38
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8938de35115b378936e3fdccd0de463d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8938de35115b378936e3fdccd0de463d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8938de35115b378936e3fdccd0de463d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1665d3dd6737255abdfb3e62c2a75dde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1665d3dd6737255abdfb3e62c2a75dde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1665d3dd6737255abdfb3e62c2a75dde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1665d3dd6737255abdfb3e62c2a75dde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1665d3dd6737255abdfb3e62c2a75dde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d5c983e6bc0b375927f5a9a7c4ebba74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([4156, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d5c983e6bc0b375927f5a9a7c4ebba74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([4156, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1665d3dd6737255abdfb3e62c2a75dde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4156, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0aaf2290134d34f1cb502ed2c21714d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(32, dtype='int32').reshape([]),
            paddle.to_tensor(32, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_3605aca3e677045f208df2206e91238b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_01421f7dff29dbb58036232da4159fa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e98a4f3b5e3959ecdeb00c7b75f915b
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_970ed0ed2c33f5ed25dae43b54386095(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0cd5fa6582c6fa47ad19ded5110ad5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 19, 29], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4042387008666992], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c507c253fd44b7c5698c341543d99e9e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 624, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_be02362f58a3e010284f5233e90262f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c507c253fd44b7c5698c341543d99e9e
    def get_inputs(self):
        return [
            paddle.uniform([1, 624, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 624, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe08d16f366db21b0fa73f21bb7dac1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3e4323a567951a3d707091b7981614b
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8938de35115b378936e3fdccd0de463d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8938de35115b378936e3fdccd0de463d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8938de35115b378936e3fdccd0de463d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8938de35115b378936e3fdccd0de463d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59945b092cb4d746030eeb6abfa865f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5e7c911d4e03e342f7799d9865d6170a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed9f40856a970d54ffe24723e19107
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70746e0deb8d731f2134be47100661ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad4d8b2af8f1e48e6c666ef31b8c2059
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_68bd62387bc532548466dbaadbf6a301(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8570defc04449c14d5bee6ebca4fed58
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d26178f57753e298d040aece04ba2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0cd5fa6582c6fa47ad19ded5110ad5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2494664192199707], dtype='float32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()