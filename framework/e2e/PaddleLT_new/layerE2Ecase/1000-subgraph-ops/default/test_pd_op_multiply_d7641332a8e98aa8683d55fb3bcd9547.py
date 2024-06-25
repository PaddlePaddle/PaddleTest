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


class TestPrimitiveOp_7494c8a3dc36dab95d8bfd7c6bb8a269(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_379f999eb4bce558d918c04cb94da334
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


class TestPrimitiveOp_84b5fca18478b592532af2a986f60ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21396504342556], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_e82e86501cf5cabc0862835cf00cdbf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b25df3b1fffbc0b25118bd310e55451
    def get_inputs(self):
        return [
            paddle.uniform([4, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.11145996302366257]], [[0.19008983671665192]], [[0.23802515864372253]], [[0.35528481006622314]]], dtype='float32').reshape([4, 1, 1]),
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


class TestPrimitiveOp_4f0bbc6e02b02733919dc5aded08172c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.23228217661380768], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_929a8c1c28a8562a35dc89fa6b9d4584(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b73d20cebb10645d031e52e745d0cf6
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.8369815945625305]], [[0.8127076029777527]], [[0.8727846145629883]], [[0.916023313999176]], [[0.8612990975379944]], [[0.8035042881965637]], [[0.9005935192108154]], [[0.8511321544647217]], [[0.8935775756835938]], [[0.8548078536987305]], [[0.8682867884635925]], [[0.8571023344993591]], [[0.9140492081642151]], [[0.8970667719841003]], [[0.9022955894470215]], [[0.9101877808570862]], [[0.8580853343009949]], [[0.9589367508888245]], [[0.8761117458343506]], [[0.921340823173523]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_5667d192a3b1fab170f36664c26fbeca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98420ee273a186f34c945b66bbab6eef
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24668194353580475]]], dtype='float32').reshape([1, 1, 1]),
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


class TestPrimitiveOp_523bc908f54c7a31f72ac488d40ce09c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.30355215072631836], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_afa14c5d91ec5c73d1c6a42a9091783c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0cd5fa6582c6fa47ad19ded5110ad5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.044028330594301224], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_0019d71e0c461271f1864149d198e389(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.19679947197437286], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_5b8defcf98be8d105abe6b2f098175bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93f34d002b99b46d05d1eb89caaa868d
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.7661869525909424]], [[0.7317941188812256]], [[0.6599379181861877]], [[0.8133224248886108]], [[0.8214729428291321]], [[0.7968551516532898]], [[0.8666788935661316]], [[0.8563156127929688]], [[0.8054947257041931]], [[0.8954908847808838]], [[0.8395293951034546]], [[0.6431409120559692]], [[0.8208305835723877]], [[0.7942108511924744]], [[0.7375274300575256]], [[0.7755582928657532]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_b6273a6b5f8f7f1d7751066f8fde6fd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.11155986785888672, -0.19301682710647583]], [[0.07615914940834045, 0.10598516464233398]], [[-0.4837867021560669, -0.07707446813583374]], [[-0.18282675743103027, 0.1370111107826233]], [[-0.35045841336250305, -0.47107231616973877]], [[-0.09866014122962952, -0.18147823214530945]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_76feda0d47ad7ab1d92d3a97ef8b293e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.05157306790351868, -0.2524963617324829]], [[-0.09184399247169495, -0.005138307809829712]], [[-0.37644314765930176, 0.1062309741973877]], [[-0.2859642505645752, 0.2382040023803711]], [[-0.07529640197753906, -0.40179216861724854]], [[0.33968156576156616, -0.29113033413887024]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_f7054c98b813f7668c6bc6722ce1d658(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.11155986785888672, -0.19301682710647583]], [[0.07615914940834045, 0.10598516464233398]], [[-0.4837867021560669, -0.07707446813583374]], [[-0.18282675743103027, 0.1370111107826233]], [[-0.35045841336250305, -0.47107231616973877]], [[-0.09866014122962952, -0.18147823214530945]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.11155986785888672, -0.19301682710647583]], [[0.07615914940834045, 0.10598516464233398]], [[-0.4837867021560669, -0.07707446813583374]], [[-0.18282675743103027, 0.1370111107826233]], [[-0.35045841336250305, -0.47107231616973877]], [[-0.09866014122962952, -0.18147823214530945]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_32b99102865b8594bdbb082b17a5a91d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f0b7cc3586001c9b2aa41c037f9740
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.05157306790351868, -0.2524963617324829]], [[-0.09184399247169495, -0.005138307809829712]], [[-0.37644314765930176, 0.1062309741973877]], [[-0.2859642505645752, 0.2382040023803711]], [[-0.07529640197753906, -0.40179216861724854]], [[0.33968156576156616, -0.29113033413887024]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.05157306790351868, -0.2524963617324829]], [[-0.09184399247169495, -0.005138307809829712]], [[-0.37644314765930176, 0.1062309741973877]], [[-0.2859642505645752, 0.2382040023803711]], [[-0.07529640197753906, -0.40179216861724854]], [[0.33968156576156616, -0.29113033413887024]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_9189fe38f52a59c139614ca5c837c5c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98420ee273a186f34c945b66bbab6eef
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.011080235242843628], [0.002222999930381775], [0.11756818741559982], [0.011925501748919487], [0.20240400731563568], [0.008813655003905296]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([[[0.13436663150787354], [0.060790590941905975], [0.4693793058395386], [0.10528217256069183], [0.06731704622507095], [0.0008048239978961647]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_58e44571b996b46fcdaefcda5f9a52ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98420ee273a186f34c945b66bbab6eef
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.01711556687951088], [0.0007783736218698323], [0.0598430335521698], [0.0515529066324234], [0.06831090152263641], [0.08953694999217987]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([[[0.13436663150787354], [0.060790590941905975], [0.4693793058395386], [0.10528217256069183], [0.06731704622507095], [0.0008048239978961647]]], dtype='float32').reshape([1, 6, 1]),
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


class TestPrimitiveOp_d2f8418a0d69503f83175b2af0a9bec9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.8743335008621216, 2.0235402584075928, 2.2377123832702637, 2.043449878692627, 2.168977737426758, 1.9117441177368164, 1.9413890838623047, 2.16556453704834, 1.886833667755127, 1.8965752124786377, 2.0499253273010254, 2.1685495376586914, 2.0631155967712402, 2.13498592376709, 1.9397042989730835, 2.0933501720428467], dtype='float32').reshape([16]),
            paddle.to_tensor([0.8585463762283325, 0.8216997385025024, 0.8039815425872803, 0.8587127923965454, 0.8389710783958435, 0.9167214035987854, 0.7072528600692749, 0.5536333322525024, 0.7418434023857117, 0.9824067950248718, 0.9695497155189514, 0.5478237867355347, 0.8586859703063965, 0.7409856915473938, 0.5669056177139282, 0.7051398754119873], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_5f27fd98e1460a7e0d7fb2ae40e64433(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.9790948629379272, 2.0919315814971924, 1.915950059890747, 2.1729745864868164, 2.163529634475708, 2.3264718055725098, 2.256845474243164, 2.2326173782348633, 1.9244996309280396, 2.2308878898620605, 2.060521364212036, 2.177788734436035, 2.0900094509124756, 1.9393761157989502, 2.2235050201416016, 2.1341819763183594], dtype='float32').reshape([16]),
            paddle.to_tensor([0.14145363867282867, 0.17830027639865875, 0.19601845741271973, 0.14128723740577698, 0.1610289216041565, 0.0832785964012146, 0.2927471101284027, 0.44636663794517517, 0.25815659761428833, 0.017593204975128174, 0.030450286343693733, 0.4521762430667877, 0.14131401479244232, 0.2590143084526062, 0.4330943524837494, 0.2948601245880127], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_a8fa562ea1e868996667000944d6a20f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4722881019115448, 0.5089336037635803, 0.5436602234840393, 0.5154375433921814, 0.542025089263916, 0.4865705072879791, 0.5084344744682312, 0.5488736629486084, 0.47413933277130127, 0.4756142199039459, 0.5125619769096375, 0.543181836605072, 0.5167289972305298, 0.5210800766944885, 0.5156542062759399, 0.5263474583625793], dtype='float32').reshape([16]),
            paddle.to_tensor([0.20642834901809692, 0.4011949598789215, 0.3455509543418884, 0.25245043635368347, 0.3066585958003998, 0.14078189432621002, 0.003124078270047903, 0.03764742612838745, 0.11097169667482376, 0.4422847628593445, 0.4226512312889099, 0.23206491768360138, 0.35839277505874634, 0.20369675755500793, 0.3251167833805084, 0.10053274035453796], dtype='float32').reshape([16]),
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


class TestPrimitiveOp_92826bc67e1c49712a1666c02e6f058f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.31752809882164], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_6b0c32837b0c10ce41432d702e02a067(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4199751615524292], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_5b08487ba8d8326f8bbea5013ee26455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b115870dac2bd4a5c9de7ce7b479b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.016477195546030998, 0.3683936297893524, 0.4820865988731384, 0.4711489975452423]]], dtype='float32').reshape([1, 1, 4]),
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


class TestPrimitiveOp_32646f911f1f1a19bd96c8573d51170f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4735390841960907], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6bb86dd9912066b9051b2d7eafcaa3cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3383793234825134], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f65063b928bacffc03f0574546d8f59b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.31494951248168945], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_1ecbf0447c620156c2aece5bb32bd6ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86a54ec7416ee4fcd7a5c436627ddd16
    def get_inputs(self):
        return [
            paddle.uniform([3, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.18307243287563324]], [[0.02208247408270836]], [[0.014094334095716476]]], dtype='float32').reshape([3, 1, 1]),
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


class TestPrimitiveOp_4e9ec0777a29977d55f410f1f0dc61e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0cd5fa6582c6fa47ad19ded5110ad5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.42615532875061035], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_cbe12668ddd3e6083a59adb6a36103d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbe12668ddd3e6083a59adb6a36103d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbe12668ddd3e6083a59adb6a36103d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbe12668ddd3e6083a59adb6a36103d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbe12668ddd3e6083a59adb6a36103d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_275fac174149c3c11a538fb12298dd5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1841, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_275fac174149c3c11a538fb12298dd5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1841, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbe12668ddd3e6083a59adb6a36103d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1fe14a1ada2bb144a12326ab62490cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_beb03874cd6e65ec170df15c443cb302(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_379f999eb4bce558d918c04cb94da334
    def get_inputs(self):
        return [
            paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[0.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_1c6910d261f8d89edc9d6fd835006885(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18493205308914185], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_a6219ef36cf0dfdda206a7da04bae05a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0cd5fa6582c6fa47ad19ded5110ad5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3982602059841156], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_3364674dce5a86025491f7fdad238b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.007246673107147217], [0.0], [0.0], [0.0], [0.0], [0.26750633120536804], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_63f7c825a3c5c2d2753d9c08fc67a779(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.27137720584869385], [-0.43975457549095154], [-0.40652668476104736], [-0.03737959265708923], [-0.008019477128982544], [0.2937753200531006], [-0.4511029124259949], [-0.16887430846691132], [-0.0783783346414566]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.10305583477020264], [0.058681756258010864], [-0.02054765820503235], [0.037331730127334595], [-0.04029268026351929], [-0.1360500454902649], [0.043546855449676514], [0.2890413999557495], [-0.14042316377162933]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_78ec883d1061c301166071fb4e41b586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1689298152923584], [-0.3045794367790222], [-0.326373815536499], [0.01093202829360962], [-0.1675238162279129], [0.369682252407074], [-0.3001260757446289], [-0.05754055827856064], [-0.3192898631095886]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.45195016264915466], [0.10231917351484299], [-0.1740318387746811], [-0.04870465397834778], [-0.016934223473072052], [-0.24927039444446564], [-0.28847536444664], [-0.1832345724105835], [0.19401900470256805]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_0398e97f2d5b915b503a9bd669a6501e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4330603778362274], [-0.3045794367790222], [-0.326373815536499], [0.01093202829360962], [0.21998120844364166], [0.3959512412548065], [-0.26420819759368896], [-0.033785074949264526], [-0.042748987674713135]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.10305583477020264], [0.29635098576545715], [0.23384928703308105], [0.3315298855304718], [0.11767389625310898], [-0.1360500454902649], [0.16240933537483215], [0.2890413999557495], [0.19401900470256805]], dtype='float32').reshape([9, 1]),
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


class TestPrimitiveOp_c1ead1999a6e19afddfdddee0af80221(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b679c4cc71d59f2e43b2dfcd94557384
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22274580597877502], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c1ead1999a6e19afddfdddee0af80221(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b679c4cc71d59f2e43b2dfcd94557384
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22274580597877502], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_c1e34ef100b7a294a00dd791a06e803e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e729ce4a58e1fb04f064fdd49605571c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.00019229290774092078], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_839bbc5e0d324d4675ac3f9e1d13e045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_839bbc5e0d324d4675ac3f9e1d13e045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_839bbc5e0d324d4675ac3f9e1d13e045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_839bbc5e0d324d4675ac3f9e1d13e045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_839bbc5e0d324d4675ac3f9e1d13e045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fbc76ab1dc0d8aa2bd2b254e89bf298f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([5562, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fbc76ab1dc0d8aa2bd2b254e89bf298f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([5562, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_839bbc5e0d324d4675ac3f9e1d13e045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_078630eb830e931a61e3a2c78ecabdb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.04207729920744896], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_3f8b9f887c9c9af3233e5eeead0bc0e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.16223658621311188], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_5927c8db70b39ef74fc9dc8e7e2ddff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.6099404096603394], dtype='float32').reshape([1]),
            paddle.to_tensor([0.1180533915758133], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_de3b1f6cc98a00c36e4792533618547c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.9359546303749084], dtype='float32').reshape([1]),
            paddle.to_tensor([0.39361587166786194], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_c75429271d10ccd065183aef2c73e211(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0cd5fa6582c6fa47ad19ded5110ad5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0832323506474495], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_88a18fec0ab3d35435862b899ebaa447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39be55716b940d949c945071172b4740
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_611131e41ae34bc471c6b68bb5efa8e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.33626288175582886], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_b036e6acf59dee53da50927d1126ade1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.15106135606765747, -0.02800169587135315, -0.07655562460422516, -0.18458950519561768, -0.10357137024402618, 0.00314900279045105], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.1507343202829361, -0.2826071083545685, -0.0314350426197052, -0.3185461759567261, -0.2002953141927719, -0.4140487313270569], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_db2f07a83fbc7b8788c25b1442c0c12e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.02277013100683689, 0.007913478650152683, 0.0024065293837338686, 0.05880028009414673, 0.02074486017227173, -0.0013038406614214182], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_2d218f14c9f257d5e061bf2334bc038e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, -0.0013038406614214182], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_b4e743c8e4b659856acf56fe4b89e306(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 0.23803360760211945, 0.20417283475399017, 0.0, 0.0, 0.007378607988357544], dtype='float32').reshape([6]),
            paddle.to_tensor([0.04888084530830383, 0.0, 0.0, 0.0, 0.41422635316848755, 0.2654237151145935], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_b41dc24c099de24203ab010e61cbfb28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.07253727316856384, 0.08720719814300537, -0.030573617666959763, -0.04275050759315491, 0.13802126049995422, 0.0461617112159729], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.1507343202829361, -0.23289291560649872, 0.0475274920463562, -0.27168169617652893, -0.2002953141927719, -0.4140487313270569], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_997ab0bb096b966f831fcfd6fd81e8b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.1538165807723999, 0.19062210619449615, 0.16335523128509521, 0.09418034553527832, 0.12180797755718231, -0.023621171712875366], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.1538165807723999, 0.19062210619449615, 0.16335523128509521, 0.09418034553527832, 0.12180797755718231, -0.023621171712875366], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_30c5ec48e3b4febf090178753b9a47db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0648808479309082, -0.1836669147014618, 0.010206416249275208, 0.07208378612995148, 0.08284178376197815, -0.045343995094299316], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0648808479309082, -0.1836669147014618, 0.010206416249275208, 0.07208378612995148, 0.08284178376197815, -0.045343995094299316], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_c57094d799086dca894a88d05ee132ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07852408289909363, 0.35324251651763916, 0.25015485286712646, 0.14183899760246277, 0.2415926307439804, 0.050391316413879395], dtype='float32').reshape([6]),
            paddle.to_tensor([0.07852408289909363, 0.35324251651763916, 0.25015485286712646, 0.14183899760246277, 0.2415926307439804, 0.050391316413879395], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_fb7e47488a357761b2a6aa8801c4ac24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.04888084530830383, 0.049714185297489166, 0.0789625346660614, 0.046864479780197144, 0.41422635316848755, 0.2654237151145935], dtype='float32').reshape([6]),
            paddle.to_tensor([0.04888084530830383, 0.049714185297489166, 0.0789625346660614, 0.046864479780197144, 0.41422635316848755, 0.2654237151145935], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_34f36a931b3ae39ae188eb9be0eee86d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5916280150413513, 0.43222132325172424, 0.23670300841331482, -0.16286125779151917, -0.14710046350955963, -0.05626258999109268], dtype='float32').reshape([6]),
            paddle.to_tensor([1.459782600402832, 1.0664626359939575, 0.5840408802032471, -0.401843786239624, -0.36295560002326965, -0.1388222873210907], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_9eb9d52497c59e11fa6b363529ee1003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.46341806650161743, 0.31551289558410645, 0.1214539185166359, 0.06142484396696091, 0.05068482458591461, 0.0077499705366790295], dtype='float32').reshape([6]),
            paddle.to_tensor([0.8636482954025269, 0.4609479010105133, 0.13824422657489777, 0.06544478237628937, 0.05339093878865242, 0.007810501381754875], dtype='float32').reshape([6]),
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


class TestPrimitiveOp_5b73b396edb4728af0ac94d1738bed07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b73b396edb4728af0ac94d1738bed07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b73b396edb4728af0ac94d1738bed07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b73b396edb4728af0ac94d1738bed07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b73b396edb4728af0ac94d1738bed07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5eace8750620c993a36997cf90ddb2e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1734, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5eace8750620c993a36997cf90ddb2e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1734, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b73b396edb4728af0ac94d1738bed07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_c3d5c54bbdc410ad1d01a18f2c65c075(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2594643235206604], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_19fcba131aedca4a68679cf8263200c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3433546721935272], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_75fa95567972135d8e8f91acc0813e97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.852647304534912, 2.228736400604248, 1.9301153421401978, 2.184696912765503, 2.1676363945007324, 2.10317063331604, 1.9660334587097168, 2.2467503547668457, 2.088048219680786, 2.2705817222595215, 2.088447093963623, 1.961663007736206, 2.150334596633911, 2.0713653564453125, 1.895040512084961, 2.1838150024414062, 2.1047916412353516, 2.1028685569763184, 1.9501138925552368, 2.219708204269409, 1.9717776775360107, 2.152820587158203, 1.9461736679077148, 2.2777602672576904], dtype='float32').reshape([24]),
            paddle.to_tensor([0.8950217366218567, 0.7568969130516052, 0.518035352230072, 0.9498879313468933, 0.8250524401664734, 0.652140200138092, 0.58144211769104, 0.648107647895813, 0.9417387247085571, 0.803348958492279, 0.6799217462539673, 0.6201160550117493, 0.7912377119064331, 0.5442841053009033, 0.9123243093490601, 0.9987372756004333, 0.7201043367385864, 0.5656696557998657, 0.5294526219367981, 0.9877932071685791, 0.8323912620544434, 0.8308432698249817, 0.5442787408828735, 0.6384990215301514], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_9025d60eb66c6f66519056bcbd38fb55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.134030818939209, 1.9788254499435425, 2.121476888656616, 2.09303617477417, 2.2146975994110107, 2.030524492263794, 2.2947871685028076, 2.1736016273498535, 1.8894554376602173, 2.148502826690674, 2.199521780014038, 2.1899044513702393, 2.022280693054199, 2.0269625186920166, 2.1642162799835205, 1.8813749551773071, 2.0201313495635986, 2.178270101547241, 2.1786949634552, 1.9380238056182861, 2.170856475830078, 2.0915040969848633, 2.201955556869507, 1.910400629043579], dtype='float32').reshape([24]),
            paddle.to_tensor([0.10497827082872391, 0.24310307204723358, 0.481964647769928, 0.0501120500266552, 0.17494754493236542, 0.34785979986190796, 0.41855791211128235, 0.3518923223018646, 0.05826127529144287, 0.19665104150772095, 0.3200782537460327, 0.37988394498825073, 0.2087623029947281, 0.45571592450141907, 0.08767567574977875, 0.0012626966927200556, 0.2798956632614136, 0.43433037400245667, 0.4705473780632019, 0.01220680121332407, 0.16760872304439545, 0.1691567450761795, 0.4557212293148041, 0.361501008272171], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_1a59197a27a6acef199164db45647e52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4705466032028198, 0.5419955849647522, 0.5055862069129944, 0.5450258851051331, 0.5439673662185669, 0.519474983215332, 0.5259090065956116, 0.5552524328231812, 0.5191195011138916, 0.5616437196731567, 0.5309998989105225, 0.5120920538902283, 0.5309004187583923, 0.5127825736999512, 0.47966015338897705, 0.5458582639694214, 0.5202739238739014, 0.5339044332504272, 0.5144180059432983, 0.5540674328804016, 0.5012862682342529, 0.5356121063232422, 0.5156847238540649, 0.5362398624420166], dtype='float32').reshape([24]),
            paddle.to_tensor([0.04863632097840309, 0.3081098794937134, 0.20273564755916595, 0.09708862751722336, 0.4469858705997467, 0.019604675471782684, 0.13594087958335876, 0.3941255807876587, 0.10889603942632675, 0.41887345910072327, 0.45694291591644287, 0.06913673877716064, 0.0006186143145896494, 0.4845738708972931, 0.4676983952522278, 0.19106808304786682, 0.008878355845808983, 0.3502928614616394, 0.1859866827726364, 0.4024882912635803, 0.08561940491199493, 0.31257855892181396, 0.3096672296524048, 0.3776283264160156], dtype='float32').reshape([24]),
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


class TestPrimitiveOp_d8f29f8957726df160b02fcc5d8dbc9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d8f29f8957726df160b02fcc5d8dbc9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d8f29f8957726df160b02fcc5d8dbc9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d8f29f8957726df160b02fcc5d8dbc9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d8f29f8957726df160b02fcc5d8dbc9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53535ea8163e57b4fc3ca78600e38832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1541, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53535ea8163e57b4fc3ca78600e38832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1541, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d8f29f8957726df160b02fcc5d8dbc9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_621957d17725b3ce0f9316664e6be436(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98420ee273a186f34c945b66bbab6eef
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.2478945553302765], [0.24430182576179504]]], dtype='float32').reshape([1, 2, 1]),
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


class TestPrimitiveOp_970ea405998363c6353faa30a6b80598(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.34481143951416016], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_5976a2a4af7733b4bf9f8cea96dc1795(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.135683536529541, 2.0193419456481934, 2.261253595352173, 2.101269006729126], dtype='float32').reshape([4]),
            paddle.to_tensor([0.792586624622345, 0.7582928538322449, 0.8323110938072205, 0.9417514204978943], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_483fd49d44d3d3a686babc41a4cc6af9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.0067954063415527, 2.207275867462158, 2.0612542629241943, 2.1883316040039062], dtype='float32').reshape([4]),
            paddle.to_tensor([0.20741339027881622, 0.24170714616775513, 0.16768890619277954, 0.05824856087565422], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_170a0f63bfda5fbc3d6212e1b7287f1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.527237594127655, 0.5161917209625244, 0.5569289922714233, 0.526585042476654], dtype='float32').reshape([4]),
            paddle.to_tensor([0.1991676539182663, 0.3436453640460968, 0.18732315301895142, 0.47298282384872437], dtype='float32').reshape([4]),
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


class TestPrimitiveOp_ef542a323dfb0b2ae7186303fb366e1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a423e03901bccbf4130859edcc903a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3754710555076599], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_985d83a2ab79a3f88679d204e52dda7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0397280752658844]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.096854567527771]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_ae0a0208d28c76493239f3cf69cabe37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2856804132461548]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.015243016183376312]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_9db649149e6651044db284112dca9b19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0397280752658844]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.21698598563671112]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_f806701357b7f72ee71db5f39236ad4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6f4f2695ad16c7c77e69861404b436
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b89266b15c2758e12034250b42f26e9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.1431642770767212]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_f2dea0cb7ca29f8b470621bf3d631361(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.007858037948608398], [-0.21845291554927826], [-0.24938222765922546], [-0.36716222763061523], [-0.009256333112716675], [0.0605430081486702]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.03684327006340027], [-0.10194528102874756], [-0.24584855139255524], [0.2912706136703491], [0.08810488879680634], [0.18930749595165253]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_6531b6e44889be61a6312aed694cdfb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.26905885338783264], [0.1727486550807953], [0.17434851825237274], [-0.00816461443901062], [-0.03874947130680084], [0.12619177997112274]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.36969172954559326], [0.09753653407096863], [-0.008594870567321777], [-0.1609843373298645], [0.05744302272796631], [0.1608583778142929]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_4829513935ed99a8097fa8af351bba29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.29271483421325684], [0.1727486550807953], [0.17434851825237274], [-0.00816461443901062], [-0.009256333112716675], [0.23934146761894226]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.03684327006340027], [0.11271576583385468], [0.028271570801734924], [0.2912706136703491], [0.19695539772510529], [0.20700159668922424]], dtype='float32').reshape([6, 1]),
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


class TestPrimitiveOp_368ce2618e4da4d9657b6e2b5af02bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93f34d002b99b46d05d1eb89caaa868d
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.7614617347717285]], [[0.7859480977058411]], [[0.9531387090682983]], [[0.872313380241394]], [[0.8555876016616821]], [[0.7153466939926147]], [[1.0]], [[0.7609117031097412]], [[0.8324828743934631]], [[0.6827892065048218]], [[0.8237350583076477]], [[0.8910555243492126]], [[0.7832010388374329]], [[0.7950375080108643]], [[0.6975187063217163]], [[0.9009041786193848]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_5ca874b0bd24fb020e14732cad8d6586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0cd5fa6582c6fa47ad19ded5110ad5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.40730491280555725], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_5f3a49dc3c6789cc76343e22f4a2ad36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98420ee273a186f34c945b66bbab6eef
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24820925295352936]]], dtype='float32').reshape([1, 1, 1]),
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


class TestPrimitiveOp_467b94d52ac7a7cf1f0bd9a5de97c8b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.06617879867553711], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_053dd41ebcd8ddcf139f7d6687a8694a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_053dd41ebcd8ddcf139f7d6687a8694a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_053dd41ebcd8ddcf139f7d6687a8694a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_053dd41ebcd8ddcf139f7d6687a8694a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_053dd41ebcd8ddcf139f7d6687a8694a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f924bcb28998dacc29b01336e519e9b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([2061, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f924bcb28998dacc29b01336e519e9b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([2061, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_053dd41ebcd8ddcf139f7d6687a8694a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89038fee494c05b5c4b527f45ddf9fe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42e5cbf8636ab2328cb24d25e02d55a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8df7bc4b250d0769e0f965a30d0513a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7682612538337708], dtype='float32').reshape([1]),
            paddle.to_tensor([0.24109217524528503], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_15fc0601cf4a9f13411e640434c52bf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8390885591506958], dtype='float32').reshape([1]),
            paddle.to_tensor([0.28952065110206604], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ddb785615a5f28de71d22cd8d609cbd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7927554845809937], dtype='float32').reshape([1]),
            paddle.to_tensor([0.4170323610305786], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cfdb2e6e29c5f275465c4b019423d563(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.6179125308990479], dtype='float32').reshape([1]),
            paddle.to_tensor([0.415785551071167], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_75cb310998b38ae40bb0e3470148d22a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.9183169007301331], dtype='float32').reshape([1]),
            paddle.to_tensor([0.47524890303611755], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6a9966a28ad5fac315a956934adb68d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8201611638069153], dtype='float32').reshape([1]),
            paddle.to_tensor([0.0366491936147213], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_316458fc82f12605fa71dcd5323afd23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7488754391670227], dtype='float32').reshape([1]),
            paddle.to_tensor([0.3872573673725128], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b79140263c62c08e4dc3601c29abfe00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7757683992385864], dtype='float32').reshape([1]),
            paddle.to_tensor([0.1179831326007843], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d86e14e07152d12a84afefa0f2acd5f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96103fdfa05cd6136dd0c5e46553a36b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7617010474205017], dtype='float32').reshape([1]),
            paddle.to_tensor([0.3960300385951996], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_b08f0dbf06b150ca2d6f130adfc691f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b679c4cc71d59f2e43b2dfcd94557384
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.20659276843070984], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b08f0dbf06b150ca2d6f130adfc691f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b679c4cc71d59f2e43b2dfcd94557384
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.20659276843070984], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2b8a78a3abc7c4452be179435e14f690(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fe80bf45124ca8664ea8278252b5644
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75b8367179e2da1e5f7b7c44f4cce475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e729ce4a58e1fb04f064fdd49605571c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.16409702599048615], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_e608741546369153ab127ae5b8673833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e608741546369153ab127ae5b8673833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e608741546369153ab127ae5b8673833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e608741546369153ab127ae5b8673833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e608741546369153ab127ae5b8673833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ab5ebdccc8b829d1c93169b231bb527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ab5ebdccc8b829d1c93169b231bb527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e608741546369153ab127ae5b8673833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_bd7d1834ad94f4bbb4732e41b9e69513(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e63b527e1fd25c01c06029d1b7a69602
    def get_inputs(self):
        return [
            paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.0792165994644165]], [[0.35136958956718445]], [[0.28345412015914917]], [[0.4633007049560547]], [[0.2462862879037857]], [[0.4422767162322998]]], dtype='float32').reshape([6, 1, 1]),
        ]


class TestPrimitiveOp_f345436a6de7eb6761742ce621005ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42e5cbf8636ab2328cb24d25e02d55a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b77a9adcbc7de3657c9b2bff7fbd4a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b77a9adcbc7de3657c9b2bff7fbd4a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b77a9adcbc7de3657c9b2bff7fbd4a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b77a9adcbc7de3657c9b2bff7fbd4a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b77a9adcbc7de3657c9b2bff7fbd4a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d169caf00208405e88a2a3399075a9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d169caf00208405e88a2a3399075a9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b77a9adcbc7de3657c9b2bff7fbd4a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_5603d1ac929f7aed62a01224c913bb7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.23294268548488617], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_a8d3f3fc466d9d384f63bebdfcc51ce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_def30c9b7633e236a259f47a1624da77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.253549724817276], [0.12049655616283417], [-0.023322254419326782], [0.1341712772846222], [0.13207146525382996]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.09657534956932068], [-0.01907147467136383], [-0.22238273918628693], [-0.28139835596084595], [-0.17056933045387268]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_3ed9720eb298b13bd08027d393b98e79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.007691273000091314], [-0.05258602648973465], [-0.16925214231014252], [-0.07432708144187927], [-0.16166751086711884]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.21568427979946136], [-0.17892664670944214], [-0.33759501576423645], [0.2636294960975647], [0.115968257188797]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_48641562c60a1291094927febb43eb52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.402584433555603], [0.1258813738822937], [-0.023077845573425293], [0.1341712772846222], [0.13207146525382996]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.006089448928833008], [-0.01907147467136383], [-0.22238273918628693], [0.2636294960975647], [0.115968257188797]], dtype='float32').reshape([5, 1]),
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


class TestPrimitiveOp_c0f9239e40ab481fcc396fa2a4967ec0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0cd5fa6582c6fa47ad19ded5110ad5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 76, 116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4849720597267151], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_670f2e6931a4bb02a35bc1152ca0a037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a423e03901bccbf4130859edcc903a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1111835464835167], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_7b650abb356973df63303b102efed8ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b679c4cc71d59f2e43b2dfcd94557384
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1536455899477005], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7b650abb356973df63303b102efed8ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b679c4cc71d59f2e43b2dfcd94557384
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1536455899477005], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0fb5684e1768a44747951f83105ec7f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fe80bf45124ca8664ea8278252b5644
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5dd9d2170da82f473bab6033f0020137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e729ce4a58e1fb04f064fdd49605571c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.44296854734420776], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_14c8ffe81a9ce0b6440f4d425ea58a80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.061662591993808746], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_05750e250a347bf860b5d5e228ae11d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05750e250a347bf860b5d5e228ae11d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05750e250a347bf860b5d5e228ae11d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05750e250a347bf860b5d5e228ae11d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05750e250a347bf860b5d5e228ae11d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6d71184808b2260584d4a67fddd35af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([2369, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b6d71184808b2260584d4a67fddd35af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([2369, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05750e250a347bf860b5d5e228ae11d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_d94d2d42ad4c2bf086a296148ae4a4a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d94d2d42ad4c2bf086a296148ae4a4a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d94d2d42ad4c2bf086a296148ae4a4a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d94d2d42ad4c2bf086a296148ae4a4a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d94d2d42ad4c2bf086a296148ae4a4a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1ec96a66f27538fbb78771e2f6c9f9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([3054, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1ec96a66f27538fbb78771e2f6c9f9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([3054, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d94d2d42ad4c2bf086a296148ae4a4a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb33d66ac1641673becbbdc1c6db98c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb33d66ac1641673becbbdc1c6db98c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb33d66ac1641673becbbdc1c6db98c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb33d66ac1641673becbbdc1c6db98c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb33d66ac1641673becbbdc1c6db98c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c4f52773e6f962aab8c6143172d80c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([3819, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c4f52773e6f962aab8c6143172d80c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([3819, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb33d66ac1641673becbbdc1c6db98c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d4782f65251f4da752e1f7fcca2d2bc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b679c4cc71d59f2e43b2dfcd94557384
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1692831814289093], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d4782f65251f4da752e1f7fcca2d2bc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b679c4cc71d59f2e43b2dfcd94557384
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1692831814289093], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_3ac8245bfd4c58bc609b7e89307a0a92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fe80bf45124ca8664ea8278252b5644
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5f4786e14183fc3d50ee8eb10389aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e729ce4a58e1fb04f064fdd49605571c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.057580769062042236], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_1f48fae8af5d054b72df6f799ff20c26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06afa0ac8681719902f3e86d0feb1b9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.9110714197158813]], [[0.8897727727890015]], [[0.9491316676139832]], [[0.9396569728851318]], [[0.8926815390586853]], [[0.936086893081665]], [[0.9218658804893494]], [[0.9558250904083252]], [[0.8945561051368713]], [[0.9359187483787537]], [[0.9453573822975159]], [[0.9548767805099487]], [[0.8521904349327087]], [[0.7774996161460876]], [[0.9180113673210144]], [[0.8945522308349609]], [[0.8467656970024109]], [[0.9145370721817017]], [[0.8750278949737549]], [[0.9656716585159302]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_00af904f91f3e0051f9befed137433a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_379f999eb4bce558d918c04cb94da334
    def get_inputs(self):
        return [
            paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


class TestPrimitiveOp_cd748197359b06d350b910844942734f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a423e03901bccbf4130859edcc903a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20729821920394897], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_ed53b229c281ae41a1a7e6223aab231c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.04416276514530182], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_cb3a6de0d8c4bc097333ac5240267a9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc92fca6b0d576aa7e787ec52e99bc7c
    def get_inputs(self):
        return [
            paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.33483582735061646]], [[0.24652710556983948]]], dtype='float32').reshape([2, 1, 1]),
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


class TestPrimitiveOp_1124428198f1be9173dfb324f0d8326c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.042593002319336, 2.327192783355713, 2.0638253688812256, 2.108158588409424, 2.2466282844543457, 2.025109052658081, 1.869448184967041, 2.3129947185516357, 1.9111738204956055, 1.9322142601013184, 2.141345739364624, 2.1675944328308105, 2.239560127258301, 2.2838408946990967, 1.9605650901794434, 2.0430891513824463, 2.2372374534606934, 2.0169076919555664, 2.188708782196045, 2.045884847640991], dtype='float32').reshape([20]),
            paddle.to_tensor([0.9400670528411865, 0.83907550573349, 0.5511564612388611, 0.995710551738739, 0.8382306694984436, 0.935483455657959, 0.7313075661659241, 0.8794151544570923, 0.8678823709487915, 0.6320870518684387, 0.7475765943527222, 0.7697842121124268, 0.701543927192688, 0.5498038530349731, 0.5099180936813354, 0.982010543346405, 0.5872577428817749, 0.7503877282142639, 0.804875373840332, 0.529133141040802], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_1881af2898ac03ea08a0ed55b0e9f589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.145280122756958, 2.0991663932800293, 1.9576224088668823, 2.1046853065490723, 2.0412521362304688, 2.015265941619873, 2.161036252975464, 1.86943781375885, 2.158451795578003, 2.0824215412139893, 2.2069036960601807, 1.862311840057373, 1.8938204050064087, 2.0171144008636475, 2.1599581241607666, 2.2342026233673096, 1.9857749938964844, 2.2292065620422363, 1.876267910003662, 2.1061301231384277], dtype='float32').reshape([20]),
            paddle.to_tensor([0.059932973235845566, 0.1609245091676712, 0.4488435387611389, 0.004289453383535147, 0.1617693454027176, 0.06451651453971863, 0.2686924338340759, 0.12058482319116592, 0.1321176439523697, 0.3679129481315613, 0.2524234354496002, 0.23021575808525085, 0.298456072807312, 0.45019611716270447, 0.49008193612098694, 0.017989452928304672, 0.4127422571182251, 0.24961227178573608, 0.19512465596199036, 0.470866858959198], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_ac257da7830f4ec3ab81b9337217306b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5121868252754211, 0.5726244449615479, 0.5040391683578491, 0.5270359516143799, 0.5533512234687805, 0.5061184763908386, 0.4869489073753357, 0.5648770928382874, 0.4859609305858612, 0.49686938524246216, 0.5394735336303711, 0.524328351020813, 0.5340930223464966, 0.5409404039382935, 0.5145710110664368, 0.5116317868232727, 0.533362090587616, 0.5174750089645386, 0.53193598985672, 0.5185630917549133], dtype='float32').reshape([20]),
            paddle.to_tensor([0.24460193514823914, 0.2284226268529892, 0.030154645442962646, 0.17981913685798645, 0.13722747564315796, 0.1968625783920288, 0.46955496072769165, 0.34556815028190613, 0.3510408103466034, 0.24403053522109985, 0.15191878378391266, 0.015035003423690796, 0.4349971115589142, 0.2135375440120697, 0.48263734579086304, 0.00949772261083126, 0.2953033745288849, 0.4858364462852478, 0.2277623564004898, 0.1553914099931717], dtype='float32').reshape([20]),
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


class TestPrimitiveOp_4abf27a309b62a5fb22ca6dbf156ac3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b73d20cebb10645d031e52e745d0cf6
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.9113849997520447]], [[0.9447422027587891]], [[0.9042185544967651]], [[0.9466654658317566]], [[0.9141700267791748]], [[0.8238815665245056]], [[0.9673439264297485]], [[0.7470681667327881]], [[0.9344660043716431]], [[0.9073324799537659]], [[0.9377887845039368]], [[0.9419918656349182]], [[0.8938634395599365]], [[0.8884957432746887]], [[0.8754721879959106]], [[0.9547819495201111]], [[0.8929009437561035]], [[0.8944354057312012]], [[0.9002806544303894]], [[0.8472577929496765]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_6af6dfc25cda5e816f34ec9f100157ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.03776758909225464], [0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.21245698630809784], [0.0], [0.1316526234149933]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_75e0740ec2d1080875d747f2478e04f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.31517642736434937], [0.06374594569206238], [0.03776758909225464], [-0.09134560823440552]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.1343723088502884], [0.33530622720718384], [0.17124499380588531], [0.32590606808662415]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_5a545ecb5b616d88b7d5432fe9c9313f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.21264928579330444], [0.018381595611572266], [0.2974410653114319], [0.38133665919303894]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.26930439472198486], [0.31753677129745483], [0.027998268604278564], [0.21617110073566437]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_7e35dad4811e6eac69adee7f3f1bce3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.31517642736434937], [0.43182724714279175], [0.2974410653114319], [0.38133665919303894]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.1221892386674881], [0.44038599729537964], [0.23443172872066498], [0.4104245603084564]], dtype='float32').reshape([4, 1]),
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


class TestPrimitiveOp_9d856d5179816d9b41e683663dc26b1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d856d5179816d9b41e683663dc26b1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d856d5179816d9b41e683663dc26b1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d856d5179816d9b41e683663dc26b1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d856d5179816d9b41e683663dc26b1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4721ccc13f8c22f859f11a12f6b1c43a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([2092, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4721ccc13f8c22f859f11a12f6b1c43a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([2092, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d856d5179816d9b41e683663dc26b1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_861df65c76a14dded08e1608b8111bca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a423e03901bccbf4130859edcc903a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.47288334369659424], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_41f8bb8b58134666ca5fd8c867c0a6e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b679c4cc71d59f2e43b2dfcd94557384
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4527304172515869], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_41f8bb8b58134666ca5fd8c867c0a6e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b679c4cc71d59f2e43b2dfcd94557384
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4527304172515869], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cbab3b741b6ad3ef14d5d0b1eefec635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fe80bf45124ca8664ea8278252b5644
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_929dd9fe739df2eab453894cc8ceedf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e729ce4a58e1fb04f064fdd49605571c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.03847096115350723], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_f6c274f1dd314ab080b6b595c3300233(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0cd5fa6582c6fa47ad19ded5110ad5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 38, 58], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10594996064901352], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_80fa5dc20d253464eccf76066b5b39a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1376037895679474], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_84c9213ae3a8d152f982d986aca08a04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0cd5fa6582c6fa47ad19ded5110ad5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2418663501739502], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_6bf28dc9f38071b0d693d361035fa1db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.43229007720947266], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_7de0fb4944880569a7ff5ceccd29cc52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2998311ba2fef68a6646e8cabf77a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.38622987270355225], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_f3be135227190629cfcade39bd1504b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3be135227190629cfcade39bd1504b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3be135227190629cfcade39bd1504b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3be135227190629cfcade39bd1504b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3be135227190629cfcade39bd1504b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba9a4237079aace3be3d6163b60ab6e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([4214, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba9a4237079aace3be3d6163b60ab6e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([4214, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3be135227190629cfcade39bd1504b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_17db703267da83908e334d9c3873f0c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0cd5fa6582c6fa47ad19ded5110ad5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 19, 29], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3334348797798157], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_c7122d8b87ed19b29c7db5a7534a7365(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0cd5fa6582c6fa47ad19ded5110ad5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3870808780193329], dtype='float32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()