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



class PrimitiveOp_ea3ad6a4af220e8de872eaa59cebe711(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_815e72764af3389ec9d9aaca8c653179(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea3ad6a4af220e8de872eaa59cebe711
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_071ed06fac5df79e7278f63bd3047a99(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21504, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 21504, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_321eeeb14550894084716310491220d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_071ed06fac5df79e7278f63bd3047a99
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21504, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_321eeeb14550894084716310491220d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_071ed06fac5df79e7278f63bd3047a99
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21504, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c6c0aca5ce2e13946369cfb6c387e262(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 576, 10, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 576, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_35768935fcadb93d1e7851377761f198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6c0aca5ce2e13946369cfb6c387e262
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e3efe362643d10e20d0d525ff9c7618e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_81eade7c0d58dbad6a8b6aea5d8ea22b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3efe362643d10e20d0d525ff9c7618e
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8736aa72e7a4f74c9b56662baf01338a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 92, 40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 92, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca6c4bbd8e0fa57a11a4141a700ad183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8736aa72e7a4f74c9b56662baf01338a
    def get_inputs(self):
        return [
            paddle.uniform([1, 92, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 92, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a86912fba1468a4cc752c058a2e18296(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 152, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3d224fa933e8f47342177fc12f72829f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a86912fba1468a4cc752c058a2e18296
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 152, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9be192a628edefe956fdca020ebee958(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 160, 30, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97bff6de71f41d9194ab02b30ccfad3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9be192a628edefe956fdca020ebee958
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_092fc4de7b2e759352aca979a690cfef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 192, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_03069ae5716b03fe4bf774259442359e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092fc4de7b2e759352aca979a690cfef
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[0.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class PrimitiveOp_8e627e1149ad0833fb50dc5f6837362b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 13, 13], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1bd472cb4494eebcaf2fc890149558c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e627e1149ad0833fb50dc5f6837362b
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_49cc3bcf5fe9bc7fdf0f0c1fbf48a019(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 16, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_116dd21e6a6fb642a1fd7ebeab1b4339(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49cc3bcf5fe9bc7fdf0f0c1fbf48a019
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_37d431d8d12ac49415bb9099f8f1b721(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 672, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0a2dd7a0b915925a451e2c1c20f74b63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37d431d8d12ac49415bb9099f8f1b721
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e6091199a470ab6f8f7a4e1eee4c94ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 32, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a5549940ac1e88c9fc7a09d4ff0bf28f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6091199a470ab6f8f7a4e1eee4c94ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_376fde9d5c241f7d14927bb3fdda5fff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 112, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0adea14173767e8441cdfa2e21183e2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_376fde9d5c241f7d14927bb3fdda5fff
    def get_inputs(self):
        return [
            paddle.uniform([43, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_123c4c37f31fa937c21ca370a3424f4b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 20, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5573d4e60d2fe9a825840ea3b70fc455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123c4c37f31fa937c21ca370a3424f4b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14353792369365692], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2dbec6386880e81321107620b7579f23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e370d55376972b0fb03a0b4ec76cc984(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2dbec6386880e81321107620b7579f23
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ffabbc4f4d7bfe414d05eb9696e61e7a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 44, 44], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_16b13955d8777246154e1326cbe30cc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffabbc4f4d7bfe414d05eb9696e61e7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8ecfa147eb81fe4e6c4ac0a05c00bd6b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6e9f507f3a5f084ac86d039e52b342c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ecfa147eb81fe4e6c4ac0a05c00bd6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e9f507f3a5f084ac86d039e52b342c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ecfa147eb81fe4e6c4ac0a05c00bd6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e9f507f3a5f084ac86d039e52b342c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ecfa147eb81fe4e6c4ac0a05c00bd6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7362f7afa42952ba39dab202622b7c5e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 336, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 336, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_96ca001d677c9e447f8014b72fed674a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7362f7afa42952ba39dab202622b7c5e
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


class PrimitiveOp_412d348f8b602df6f1332b51a43c0549(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b101b197b9942cadee0365e2b6b533a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_412d348f8b602df6f1332b51a43c0549
    def get_inputs(self):
        return [
            paddle.uniform([4, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.20390234887599945]], [[0.2761944532394409]], [[0.44851556420326233]], [[0.4557492733001709]]], dtype='float32').reshape([4, 1, 1]),
        ]


class PrimitiveOp_5d2f5611870d3f1d8af810feb55e461d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e294f50de410d34a91184777073d8595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2f5611870d3f1d8af810feb55e461d
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f87f6bf733fc6abbb4e44c5c18dd24a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 104, 104], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f7634b0ab89874212623adaae8f3cfe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f87f6bf733fc6abbb4e44c5c18dd24a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_baeabc11c986938180f5901cbd5e2eb1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1024, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cd888c6723646fbd0a56514ea60843ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_baeabc11c986938180f5901cbd5e2eb1
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


class PrimitiveOp_edab10024c1371104d300a259348b016(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_79e4a4f0c96996fd0e6dcc149644e70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edab10024c1371104d300a259348b016
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8d467e8235e880f3c5a8f5baf9bacd88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_225b7f6722f0bb79f34104a9f49996d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d467e8235e880f3c5a8f5baf9bacd88
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.47636646032333374], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_97eab3f01facdd16e7808d2e9a26d546(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c293871a50f709ae77bd5ac416e7a68e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97eab3f01facdd16e7808d2e9a26d546
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c293871a50f709ae77bd5ac416e7a68e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97eab3f01facdd16e7808d2e9a26d546
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c293871a50f709ae77bd5ac416e7a68e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97eab3f01facdd16e7808d2e9a26d546
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_163ad744ee292963f38996c7c6abc563(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06afa0ac8681719902f3e86d0feb1b9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.8803272843360901]], [[0.92710280418396]], [[0.8239691257476807]], [[0.947374165058136]], [[0.8867131471633911]], [[0.9509221315383911]], [[0.9507958889007568]], [[0.8666858077049255]], [[0.8413794040679932]], [[0.8063572645187378]], [[0.9437206983566284]], [[0.872735857963562]], [[0.8971161842346191]], [[0.9170681834220886]], [[0.9061281085014343]], [[0.8637533187866211]], [[0.8753122091293335]], [[0.8761754631996155]], [[0.8924030065536499]], [[0.9468423128128052]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class PrimitiveOp_340311d4fd3d640fd702110a039dfe0a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 17, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d015130f9fc6d937009e3b3ed5bcb0db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_340311d4fd3d640fd702110a039dfe0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bc3ba840de278f277d1bfb295374b94e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 10, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b9d2e37fe24c680ee097e34fe6fda963(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc3ba840de278f277d1bfb295374b94e
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5bf64d7cf8054186bcc7796217057d48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d3af070f07a4f4891f3ed3f0563cedc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bf64d7cf8054186bcc7796217057d48
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_72ea59c5a34ae0ce732cb715c17596ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 2100], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 2100], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02984a867492da0442f1807e77125247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72ea59c5a34ae0ce732cb715c17596ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02984a867492da0442f1807e77125247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72ea59c5a34ae0ce732cb715c17596ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f226bf699b81a8340f1c4c0f15e3c537(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 2100], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_48a76272d52e1f27b04762ac341ec718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f226bf699b81a8340f1c4c0f15e3c537
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24856245517730713]]], dtype='float32').reshape([1, 1, 1]),
        ]


class PrimitiveOp_3bd07c3477ed51e2e89c59143374aa6a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2100, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_93fcc00ae37b7e492d6bfe053b6eac73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bd07c3477ed51e2e89c59143374aa6a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2100, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f82bae9f01017bc84c73b0afb0ceb402(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4f79157a9503b867a883a39a865d1e32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f82bae9f01017bc84c73b0afb0ceb402
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.022010652348399162], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7292c006e54e0195a64c3cccb3a02fb4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4096, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[4096, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_19f70bcf1a3b09a3eb018ff6782ea67f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7292c006e54e0195a64c3cccb3a02fb4
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d07132c8b6e809630ba97c8e007c0d37(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 240, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cbbe62f8db8441233de6fbcdd5240f62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d07132c8b6e809630ba97c8e007c0d37
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ca583380b0e23044064e1e54c702ddfa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 60, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 60, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e30ad14136c518d6c57ce24dc8389fc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca583380b0e23044064e1e54c702ddfa
    def get_inputs(self):
        return [
            paddle.uniform([10, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_373f28edd813211622ed1dd23f15b0b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3800, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[3800, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b03ffb5ee15dafa65de3266a43786ab2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_373f28edd813211622ed1dd23f15b0b4
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_46b4af2e724b4b2010f490551ae44b0a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_58a379cc4e133b750eab41f9ab80825c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b4af2e724b4b2010f490551ae44b0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58a379cc4e133b750eab41f9ab80825c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b4af2e724b4b2010f490551ae44b0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58a379cc4e133b750eab41f9ab80825c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b4af2e724b4b2010f490551ae44b0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58a379cc4e133b750eab41f9ab80825c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b4af2e724b4b2010f490551ae44b0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4245913426c50e23e26e996c4e3e55d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 672, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97eb22375d38f1d6c87856e02c2f16c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4245913426c50e23e26e996c4e3e55d4
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4c69efc00629a6fa23d714fdb6e006e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a5af04361edca3e1efb4d84b2d0d72e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c69efc00629a6fa23d714fdb6e006e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a5af04361edca3e1efb4d84b2d0d72e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c69efc00629a6fa23d714fdb6e006e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a5af04361edca3e1efb4d84b2d0d72e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c69efc00629a6fa23d714fdb6e006e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e4b66f6662af9bd1d97588954ac2d0fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 44, 44], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8424fd8c9c30c9e20c6dd09446f9381d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4b66f6662af9bd1d97588954ac2d0fc
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5067f5666ed30cfdbbb90847d8b3066a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_981b2761af299bde8581549db98234de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5067f5666ed30cfdbbb90847d8b3066a
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_acb75ac7d8898f56f550b9ebe95dc4b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 240, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5db086effc395232daca9ffccee7db80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acb75ac7d8898f56f550b9ebe95dc4b7
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e8b788cb6625f4842bd8dcca7be8c085(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 1152, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 1152, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7f957723d08a914a51f72ce7c3c47dc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8b788cb6625f4842bd8dcca7be8c085
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1302bca5f85a2e2cee7fc197723ff908(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 48, 48], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4844804280a1880b68cbe8b1302d7bb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1302bca5f85a2e2cee7fc197723ff908
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9eaf54001e5dd18b6a49a167eb27dc6c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bd6042c656234cc0f208cca377c2cf98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9eaf54001e5dd18b6a49a167eb27dc6c
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4d22fc4b89bad4a9003a61e438ce2bf4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[150, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[150, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_73944ba64cf65957bc8282637072133b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d22fc4b89bad4a9003a61e438ce2bf4
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_be9379a2ffd678c991404990a3f835ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a85220b52f92837d99622b6a7c05f441(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be9379a2ffd678c991404990a3f835ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14931374788284302], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4ec5685eb7ecd374ae0a21bd83d0d5c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 112, 112], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1f56334f6ae917e74b6ec88055b9fc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ec5685eb7ecd374ae0a21bd83d0d5c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f86f981c78f225e318788a1f9d153c9d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aadc422281d8fa70d45330cb5868ba49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f86f981c78f225e318788a1f9d153c9d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5a3ff77e2c3b2ffef3d39309f2e443eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 336, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[145, 336, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5bf59b4f8f55797681a7e31e0c9fe2b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a3ff77e2c3b2ffef3d39309f2e443eb
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2df90e7fe7ffdc7f4eb66c6e3d30b961(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 128, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1095d3fb2ffa7ebd7c82d25b853d012c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2df90e7fe7ffdc7f4eb66c6e3d30b961
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c9afbcc6f57ba50a3da6628142dd38ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 28, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_87e8de104ee26551fcd6138b46f611a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9afbcc6f57ba50a3da6628142dd38ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.028488636016845703], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9a5eb7010acdccc0e8dd26e43e3813f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 34, 34], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00ac1073c535c84ebf66c213b2d22014(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a5eb7010acdccc0e8dd26e43e3813f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_39670343df85f06e46d21b863131110d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 80, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e7a06e31a81033050d1d8834707303f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39670343df85f06e46d21b863131110d
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.7435420751571655]], [[0.8665276169776917]], [[0.7579473257064819]], [[0.8741436004638672]], [[0.8588787317276001]], [[0.7497829794883728]], [[0.7431142926216125]], [[0.8241839408874512]], [[0.962710976600647]], [[0.8454965949058533]], [[0.7933272123336792]], [[0.7493131160736084]], [[0.7719723582267761]], [[0.5867494344711304]], [[0.8012163639068604]], [[0.8430553078651428]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class PrimitiveOp_73ac9e1788bcb88b4f8ecb5ae2b2ae40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_55a38fc41b063d0dd3b9da14c5a67689(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73ac9e1788bcb88b4f8ecb5ae2b2ae40
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_204265cbd2e15645b84b480dd7a0290c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 336, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[145, 336, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca06286eede5e26d27a0edcc88a7d967(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204265cbd2e15645b84b480dd7a0290c
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d9ac2ea86a8d7e7133229c18b0519997(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 44, 48, 48], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 44, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e57c1fe0cd741713389284571200bea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9ac2ea86a8d7e7133229c18b0519997
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2f16b2d4d9a319dfa9e4a7b0b6bcfee2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 160, 24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dbd28292136df5a23715b356055cbc89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f16b2d4d9a319dfa9e4a7b0b6bcfee2
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_21640bfeddd1cc679a8a6fae429cb547(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 21824, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6, 1, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6a526845b140a324e64d2e96e0327f1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21640bfeddd1cc679a8a6fae429cb547
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.18047603964805603, 0.018548965454101562]], [[0.18639850616455078, 0.07878831028938293]], [[-0.27625542879104614, -0.02650339901447296]], [[0.27415549755096436, 0.3779466450214386]], [[-0.14886969327926636, -0.2244105041027069]], [[0.040308043360710144, 0.31997889280319214]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_7339f100024bb1043c2875c54a14e5c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21640bfeddd1cc679a8a6fae429cb547
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.1609879434108734, -0.06676539778709412]], [[0.041958920657634735, -0.28659969568252563]], [[0.16202175617218018, 0.20733529329299927]], [[-0.11345814913511276, 0.43742579221725464]], [[-0.026480108499526978, -0.27732908725738525]], [[0.2622227966785431, 0.29486244916915894]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class PrimitiveOp_4992024e84675fd31e5997a5c142f74e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6, 1, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53d1546147f93b0a496d59bb8d5dbe12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4992024e84675fd31e5997a5c142f74e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.18047603964805603, 0.018548965454101562]], [[0.18639850616455078, 0.07878831028938293]], [[-0.27625542879104614, -0.02650339901447296]], [[0.27415549755096436, 0.3779466450214386]], [[-0.14886969327926636, -0.2244105041027069]], [[0.040308043360710144, 0.31997889280319214]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.18047603964805603, 0.018548965454101562]], [[0.18639850616455078, 0.07878831028938293]], [[-0.27625542879104614, -0.02650339901447296]], [[0.27415549755096436, 0.3779466450214386]], [[-0.14886969327926636, -0.2244105041027069]], [[0.040308043360710144, 0.31997889280319214]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_51f2fee52cb423a0ccb9ed1669c634e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4992024e84675fd31e5997a5c142f74e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.1609879434108734, -0.06676539778709412]], [[0.041958920657634735, -0.28659969568252563]], [[0.16202175617218018, 0.20733529329299927]], [[-0.11345814913511276, 0.43742579221725464]], [[-0.026480108499526978, -0.27732908725738525]], [[0.2622227966785431, 0.29486244916915894]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.1609879434108734, -0.06676539778709412]], [[0.041958920657634735, -0.28659969568252563]], [[0.16202175617218018, 0.20733529329299927]], [[-0.11345814913511276, 0.43742579221725464]], [[-0.026480108499526978, -0.27732908725738525]], [[0.2622227966785431, 0.29486244916915894]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class PrimitiveOp_fe310264e1b88b3efe1ed272b8463698(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 6, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_315fc5c27be343c9baa25ae896e98155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe310264e1b88b3efe1ed272b8463698
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.00597178190946579], [0.008287292905151844], [0.02137475088238716], [0.10178864747285843], [0.01953021250665188], [0.03354442119598389]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([[[0.2820061147212982], [0.2551458179950714], [0.22969898581504822], [0.025023801252245903], [0.18655428290367126], [0.054804302752017975]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_17dc3c1f27f4bdff80810888f79c01a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe310264e1b88b3efe1ed272b8463698
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.005293814465403557], [0.02430202253162861], [0.018219055607914925], [0.09228444844484329], [0.021622128784656525], [0.06144017353653908]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([[[0.2820061147212982], [0.2551458179950714], [0.22969898581504822], [0.025023801252245903], [0.18655428290367126], [0.054804302752017975]]], dtype='float32').reshape([1, 6, 1]),
        ]


class PrimitiveOp_d760e43b8bcab16342b5849d4aa01b62(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_581a115768e3c7b13ce21c5b3054c571(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d760e43b8bcab16342b5849d4aa01b62
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


class PrimitiveOp_11fa09a6b5c44678748f93ef61cb709b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 11, 11], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d389b58997497715c05cf7d154c7fc51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11fa09a6b5c44678748f93ef61cb709b
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4f6cfb410f118dc64133e2d1d9b9a911(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[40, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e285c8c7b6b5fe2aab561f3f39e409db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f6cfb410f118dc64133e2d1d9b9a911
    def get_inputs(self):
        return [
            paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7f44a67f1dfb1f50ea23906280e5ff14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 13, 13], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_518e5fb018bdef5dc1a53db9c2c500b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f44a67f1dfb1f50ea23906280e5ff14
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_da5da8a21c60e75fee94a83a9ddcb386(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_46b10f3984ac4162b3ed3fd9a77f1c32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da5da8a21c60e75fee94a83a9ddcb386
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3e5b08f724933040792e48be5419527f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dfb277c4f78213e8dc59ec5a072c1864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e5b08f724933040792e48be5419527f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b24ad60e80ab24bb6b066b48623adab2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_44d7f426003eaeb0bfc01911508afb62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b24ad60e80ab24bb6b066b48623adab2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44d7f426003eaeb0bfc01911508afb62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b24ad60e80ab24bb6b066b48623adab2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44d7f426003eaeb0bfc01911508afb62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b24ad60e80ab24bb6b066b48623adab2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44d7f426003eaeb0bfc01911508afb62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b24ad60e80ab24bb6b066b48623adab2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ec38d62e74c14a10210d7629afce5c6a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 80, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1108587c65f6c46a30937405754bec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec38d62e74c14a10210d7629afce5c6a
    def get_inputs(self):
        return [
            paddle.uniform([43, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5c3d536f6cc02ffd67c2d6197f0530bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3800, 81], dtype='float32'),
            paddle.static.InputSpec(shape=[3800, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7987f3fdcc957fff68ef3322bfabf575(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c3d536f6cc02ffd67c2d6197f0530bc
    def get_inputs(self):
        return [
            paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7852bc1b6dcc7869f2cc107389bc5f0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            paddle.static.InputSpec(shape=[16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40b735f649b7fdf58fd5fe6c5d358c22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7852bc1b6dcc7869f2cc107389bc5f0b
    def get_inputs(self):
        return [
            paddle.to_tensor([1.9191997051239014, 2.213761568069458, 2.2604475021362305, 2.063676118850708, 2.0685040950775146, 1.8833965063095093, 2.2101597785949707, 2.061204433441162, 1.9800026416778564, 2.4130754470825195, 1.8956514596939087, 2.2049989700317383, 2.193824291229248, 1.891596794128418, 2.015496253967285, 1.9924213886260986], dtype='float32').reshape([16]),
            paddle.to_tensor([0.8116477131843567, 0.5907785892486572, 0.8717184066772461, 0.5726866722106934, 0.7331197261810303, 0.5746523141860962, 0.5234342813491821, 0.5551745891571045, 0.9761582016944885, 0.8448885679244995, 0.6607785224914551, 0.5640827417373657, 0.5257415771484375, 0.9752717614173889, 0.5913959741592407, 0.6230497360229492], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_086cb621e6c19d0d11fab8d6d2044edf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7852bc1b6dcc7869f2cc107389bc5f0b
    def get_inputs(self):
        return [
            paddle.to_tensor([2.3045895099639893, 2.2724123001098633, 2.03210711479187, 2.249232769012451, 2.3866655826568604, 1.96580171585083, 2.1738381385803223, 1.894432783126831, 2.1372039318084717, 2.046835422515869, 2.0252630710601807, 2.1759355068206787, 2.2417712211608887, 2.0903611183166504, 2.1964030265808105, 2.2440195083618164], dtype='float32').reshape([16]),
            paddle.to_tensor([0.1883522868156433, 0.4092213809490204, 0.1282816231250763, 0.42731329798698425, 0.26688024401664734, 0.4253476560115814, 0.47656574845314026, 0.4448254108428955, 0.023841800168156624, 0.15511144697666168, 0.3392214775085449, 0.4359172284603119, 0.4742584228515625, 0.024728234857320786, 0.40860405564308167, 0.3769502341747284], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_6576257db1845d992844ec1085f21869(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7852bc1b6dcc7869f2cc107389bc5f0b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.49794721603393555, 0.5594406723976135, 0.5577889084815979, 0.5357416868209839, 0.5383537411689758, 0.4796118140220642, 0.5482125282287598, 0.49675506353378296, 0.49593764543533325, 0.5890668630599976, 0.4849046468734741, 0.5480824112892151, 0.5541409254074097, 0.47412797808647156, 0.5223538875579834, 0.521815299987793], dtype='float32').reshape([16]),
            paddle.to_tensor([0.23322135210037231, 0.372314453125, 0.45340174436569214, 0.25258103013038635, 0.04820200428366661, 0.3095371425151825, 0.007916051894426346, 0.4476539194583893, 0.2522156834602356, 0.04861040413379669, 0.05402250215411186, 0.09698207676410675, 0.07473306357860565, 0.008590548299252987, 0.4644331932067871, 0.0025238660164177418], dtype='float32').reshape([16]),
        ]


class PrimitiveOp_3630186f00bb00d90ccb305173718830(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 240, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[145, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1f5b22d06eafff073f6abcdb0aeffee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3630186f00bb00d90ccb305173718830
    def get_inputs(self):
        return [
            paddle.uniform([145, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_210051582262298800fe8fc79767c3f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 80, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b40b042bff90d2c10cdceab0d1cfe580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_210051582262298800fe8fc79767c3f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.476359486579895], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_fd1476e44f472b1b8077de762360c50b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 60, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_40031ee5406b4c8a4ccffb1822b49a05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd1476e44f472b1b8077de762360c50b
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_32074b59bac5abaf1b8a9d10910ce304(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_77d557ba6f6a303734664ea0cf392a52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32074b59bac5abaf1b8a9d10910ce304
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_77d557ba6f6a303734664ea0cf392a52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32074b59bac5abaf1b8a9d10910ce304
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_77d557ba6f6a303734664ea0cf392a52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32074b59bac5abaf1b8a9d10910ce304
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c63c960682dda3a4cada933be990e9d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 14, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ddd90948df8f07acc37940aa8cf6890(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c63c960682dda3a4cada933be990e9d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4769103527069092], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_88524ba379499f55c3ea410fbee25873(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 300, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9929ef9e9197c63c1bab585e2d4c0268(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88524ba379499f55c3ea410fbee25873
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.3934592306613922, 0.06741175800561905, 0.3000420928001404, 0.3393118381500244]]], dtype='float32').reshape([1, 1, 4]),
        ]


class PrimitiveOp_4ba5f2f9d7280059f0268c44cc5db73b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d310e75b7518c6e98e3259b0b624cbcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ba5f2f9d7280059f0268c44cc5db73b
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ce4ace322016d97bf695286595d16834(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 22, 33], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b16c77b7df5e5b0c5192566c2ef3cb94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce4ace322016d97bf695286595d16834
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3163110911846161], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2b0bf5dc6a91df258aef9a721530f24e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 23, 35], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af43e95ec0ab5d52fbfe4a655801b517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b0bf5dc6a91df258aef9a721530f24e
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3864208459854126], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_07c059a42edcf3de218e97da9fbac96f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 46, 70], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e63174658409a6f2366fd96d01b7955(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07c059a42edcf3de218e97da9fbac96f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.12430611252784729], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6e9f507f3a5f084ac86d039e52b342c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ecfa147eb81fe4e6c4ac0a05c00bd6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e9f507f3a5f084ac86d039e52b342c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ecfa147eb81fe4e6c4ac0a05c00bd6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e9f507f3a5f084ac86d039e52b342c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ecfa147eb81fe4e6c4ac0a05c00bd6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e9f507f3a5f084ac86d039e52b342c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ecfa147eb81fe4e6c4ac0a05c00bd6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b8253ad6111de39b8930be7b782affcb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_db48a6cc79be5577a4cb6273e7aba3e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8253ad6111de39b8930be7b782affcb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db48a6cc79be5577a4cb6273e7aba3e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8253ad6111de39b8930be7b782affcb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db48a6cc79be5577a4cb6273e7aba3e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8253ad6111de39b8930be7b782affcb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b8c4d4815ed87c008011679ed4482b0f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672, 34, 34], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f2b0300169c21c3a95af770bb63986ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8c4d4815ed87c008011679ed4482b0f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c712740622f241d3603e5831e540ce30(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[3, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_45337e0a0bacf835c6c268403891ef4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c712740622f241d3603e5831e540ce30
    def get_inputs(self):
        return [
            paddle.uniform([3, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.015987955033779144]], [[0.10847919434309006]], [[0.1562148481607437]]], dtype='float32').reshape([3, 1, 1]),
        ]


class PrimitiveOp_61809abe8e2d7d7fc05391c33fd36811(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 577, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_77a0961fe73d55cc2b28f77913898154(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61809abe8e2d7d7fc05391c33fd36811
    def get_inputs(self):
        return [
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d2aa2f45786454b6fd609b9ee6d988bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[150], dtype='float32'),
            paddle.static.InputSpec(shape=[150], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0049ceff0b3a88b4e65539aa51b3297c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2aa2f45786454b6fd609b9ee6d988bf
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_304b82d9e423ebd888952055c9992f78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 60, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 60, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4f3d533c144ecb9a8435335d88fdbcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_304b82d9e423ebd888952055c9992f78
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a2dd7a0b915925a451e2c1c20f74b63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37d431d8d12ac49415bb9099f8f1b721
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a26dfc3e1ee7aecbc2a9fe0ad4767ca7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_193ce7e96e845d9ee9ffc124a8957635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a26dfc3e1ee7aecbc2a9fe0ad4767ca7
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3877077102661133], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_b73eb6d43f9b406cfe532f993fde9f7b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 30, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a6a83cd2a64c982e0e580b6880c0c26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b73eb6d43f9b406cfe532f993fde9f7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8098960eb115a79955652f745c57d4f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 22, 22], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_26032e168cdc9c2cca0df1b3c85ec3a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8098960eb115a79955652f745c57d4f2
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_66b314de6dbc0fb9bda9195c20505f5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9dfbae5baa1ecdeb4328b9b37f7802c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66b314de6dbc0fb9bda9195c20505f5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b1f1df3749cd6f1f3a332144db4ce2aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 320, 22, 22], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e7f2b81a5a25a750a090ff20b10eda50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1f1df3749cd6f1f3a332144db4ce2aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_66795651b6916f9d58c606f9f42ec042(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            paddle.static.InputSpec(shape=[40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5a470d4a7336d8e70f6d6c9a279205b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66795651b6916f9d58c606f9f42ec042
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b7e98117ef3be0fa02b989698f62029b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 872, 20, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 872, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ad472af5ab080e7cc81acff2a00e75c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7e98117ef3be0fa02b989698f62029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_83c3cdb6e829f210bfd695bbc50ad272(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6cb7e69001d0e18af310bc3823478ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83c3cdb6e829f210bfd695bbc50ad272
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cb7e69001d0e18af310bc3823478ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83c3cdb6e829f210bfd695bbc50ad272
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cb7e69001d0e18af310bc3823478ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83c3cdb6e829f210bfd695bbc50ad272
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cb7e69001d0e18af310bc3823478ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83c3cdb6e829f210bfd695bbc50ad272
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fbb9b388a9ca6cc168775c14411f09d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 100, 18, 18], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 100, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a852257cb40c7db3029bb34affc3cd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbb9b388a9ca6cc168775c14411f09d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_005762fbb1c18ab515f0079ae9d762cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_93e863ec5248bbc52aa3724467be7956(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_005762fbb1c18ab515f0079ae9d762cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_845010c17ad65723784ac612a9b05a52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1786, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1786, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aac3c2619ace9c1c55b0d94b73418b07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_845010c17ad65723784ac612a9b05a52
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aac3c2619ace9c1c55b0d94b73418b07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_845010c17ad65723784ac612a9b05a52
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aac3c2619ace9c1c55b0d94b73418b07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_845010c17ad65723784ac612a9b05a52
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aac3c2619ace9c1c55b0d94b73418b07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_845010c17ad65723784ac612a9b05a52
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aac3c2619ace9c1c55b0d94b73418b07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_845010c17ad65723784ac612a9b05a52
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4cfde35728977a161b06f75ee0de02b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1786, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1786, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80f76b802d4bfeece03ba6cc811c29f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cfde35728977a161b06f75ee0de02b2
    def get_inputs(self):
        return [
            paddle.uniform([1786, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_80f76b802d4bfeece03ba6cc811c29f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cfde35728977a161b06f75ee0de02b2
    def get_inputs(self):
        return [
            paddle.uniform([1786, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aac3c2619ace9c1c55b0d94b73418b07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_845010c17ad65723784ac612a9b05a52
    def get_inputs(self):
        return [
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1786, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b20b0336e12a00d9214d414a53a93361(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15200], dtype='float32'),
            paddle.static.InputSpec(shape=[15200], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc3d59129f81ded7d193113e1dd756a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b20b0336e12a00d9214d414a53a93361
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8fb038135fe2fa8372d9386ebb79df7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 112, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f9f8fa4b90fda6dde37e02e8b8a682b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fb038135fe2fa8372d9386ebb79df7d
    def get_inputs(self):
        return [
            paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class PrimitiveOp_c12ca22a303ef482473d6959339461cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 6, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_31c67664ba3b7ce6925fbee397f4a79c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c12ca22a303ef482473d6959339461cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3490394651889801], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5676c8cad1ef2749a80be2440823e881(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672, 22, 22], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e255ba8c14a7cd3dedef7eed0ce03243(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5676c8cad1ef2749a80be2440823e881
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_53d856c85717ca4bec7292dee1ab55f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 40, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf281686a95766847e9b75df3e35cf87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53d856c85717ca4bec7292dee1ab55f9
    def get_inputs(self):
        return [
            paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class PrimitiveOp_fe45d0674cc85e7691adafe359a25cdb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bdd73a08eb5c4c8ec9c4ac31dce06204(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe45d0674cc85e7691adafe359a25cdb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9f4316e09ac823e38388127c20d36206(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 5, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_17d8033efa002413bb60d5cbb8ca9dbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f4316e09ac823e38388127c20d36206
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.46143826842308044], dtype='float32').reshape([1]),
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


class PrimitiveOp_626ae5005bb74d0d19e833cd0b4aec1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab592917f0852e524a311a54046e3084(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626ae5005bb74d0d19e833cd0b4aec1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab592917f0852e524a311a54046e3084(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626ae5005bb74d0d19e833cd0b4aec1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab592917f0852e524a311a54046e3084(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626ae5005bb74d0d19e833cd0b4aec1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c53df8392ce8625ae243f2ee7b563a5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_653f25df76c1959f8c4ae7ccb5336e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c53df8392ce8625ae243f2ee7b563a5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_653f25df76c1959f8c4ae7ccb5336e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c53df8392ce8625ae243f2ee7b563a5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_653f25df76c1959f8c4ae7ccb5336e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c53df8392ce8625ae243f2ee7b563a5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_653f25df76c1959f8c4ae7ccb5336e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c53df8392ce8625ae243f2ee7b563a5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e14f6ebf089ccc1fb85f749a2c3c30a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 13, 13], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_33d327c83edcb062bf9ed1837bb8f874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e14f6ebf089ccc1fb85f749a2c3c30a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8272a0029c17a0981d6dc0bb8f84c77b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 336, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[171, 336, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a039062873e752db115c06d5c75f8048(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8272a0029c17a0981d6dc0bb8f84c77b
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_674a820d16d84b0c122083859b96efc3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240, 44, 44], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6466f3d5207f9c8fa327d4448d94ce24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_674a820d16d84b0c122083859b96efc3
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9755cc80340b3da84850fef2c74afd00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 24, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5a9a563d831685f7cbcd07488f0a78f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9755cc80340b3da84850fef2c74afd00
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8c2d0ef34d18b466423e4abfe815aed4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_26bf797eaa3a08b2975c27f5edec9f30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c2d0ef34d18b466423e4abfe815aed4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_535ac6c81c54fa149b44dd4f08f0cbf8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672, 11, 11], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af78a4b4ef520360b2569a1be22b835d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_535ac6c81c54fa149b44dd4f08f0cbf8
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5483c1196783a3f5fa34f740c0f96af4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 320, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de8c835b71e363d6a706ca83884d02eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5483c1196783a3f5fa34f740c0f96af4
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_13ed1ba3c28e87e79667627287243f3b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e528cf0e2f60be445b356baeceb7805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed1ba3c28e87e79667627287243f3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e528cf0e2f60be445b356baeceb7805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed1ba3c28e87e79667627287243f3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e528cf0e2f60be445b356baeceb7805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed1ba3c28e87e79667627287243f3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f833bb4f34910f3c00dfabd43cacf512(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 96, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fad9baea40a9b99db68f0f819a845b4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f833bb4f34910f3c00dfabd43cacf512
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 96, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d70b658edeaf15347771e0b56340e195(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 38, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ead3e53e84e796ca804c978bf622a103(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d70b658edeaf15347771e0b56340e195
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c11bcb80120e9156b106a01d1a033d7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672, 19, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_732366250a3cd683841bb6a8eea0087c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c11bcb80120e9156b106a01d1a033d7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d636a839c1554b6e0420f4abb46f545f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_11e808ae9b13b846a14edbfa88601359(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d636a839c1554b6e0420f4abb46f545f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11e808ae9b13b846a14edbfa88601359(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d636a839c1554b6e0420f4abb46f545f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11e808ae9b13b846a14edbfa88601359(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d636a839c1554b6e0420f4abb46f545f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11e808ae9b13b846a14edbfa88601359(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d636a839c1554b6e0420f4abb46f545f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1c9b691ab08eeb1c6a0c5324789037c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2878d356e45aafddc05eef2c7e7a4c3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c9b691ab08eeb1c6a0c5324789037c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_61e2a73b85b68cf70c72f6ba6e376320(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b15426ac8e7990c245430c804aacaf5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61e2a73b85b68cf70c72f6ba6e376320
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.16944843530654907], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.23409157991409302], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_4c59e055bf4f589ff29621cb2ad86d65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61e2a73b85b68cf70c72f6ba6e376320
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.1678384244441986], [0.18568190932273865], [-0.0950990617275238], [-0.05133241415023804], [-0.29463353753089905], [0.2958884537220001], [0.2811228930950165], [-0.09302261471748352], [-0.0060365647077560425]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.11872529983520508], [-0.4231604039669037], [0.2369188517332077], [0.12562304735183716], [0.23764827847480774], [0.2131078988313675], [-0.006825923919677734], [0.09757381677627563], [0.03660881519317627]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_13c7ca082b7404da9b31b9d2f614e93e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61e2a73b85b68cf70c72f6ba6e376320
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.40390074253082275], [0.40603211522102356], [-0.21587994694709778], [-0.050048500299453735], [0.09798401594161987], [-0.11813796311616898], [-0.10573890805244446], [-0.04216012358665466], [-0.020097553730010986]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.08044969290494919], [0.27613258361816406], [0.24830251932144165], [-0.017092138528823853], [-0.18874689936637878], [-0.3528631031513214], [0.22593224048614502], [-0.30411678552627563], [-0.2992132902145386]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_6beb64b04451f6bd4090925223c7185a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61e2a73b85b68cf70c72f6ba6e376320
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.40390074253082275], [0.42226558923721313], [-0.004615187644958496], [0.12430831789970398], [0.09798401594161987], [0.2958884537220001], [0.2811228930950165], [0.024507299065589905], [0.1669752597808838]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3942401707172394], [0.27613258361816406], [0.25112980604171753], [0.14098328351974487], [0.23764827847480774], [0.2131078988313675], [0.353200227022171], [0.09757381677627563], [0.03660881519317627]], dtype='float32').reshape([9, 1]),
        ]


class PrimitiveOp_81308db06791c5f7841fe9f2d2f68387(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 36, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bd3e54743e46a40037751d540ecedaf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81308db06791c5f7841fe9f2d2f68387
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f9e31009351162921b8651dea231e5cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 240, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_58dc33519f82a00c589e9365c4ed50f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9e31009351162921b8651dea231e5cd
    def get_inputs(self):
        return [
            paddle.uniform([10, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b73850f75e1007e6dddd8446f9d1329e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21824, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 21824, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9eabe3541f5e5fcbb100234b3b7977e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b73850f75e1007e6dddd8446f9d1329e
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c5e00ddb5250321479259e986cdd2528(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 96, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_749e7ebb28b45c42341bfa042c093e44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5e00ddb5250321479259e986cdd2528
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_979512685801d46188aeb0d32fdcdda2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 576, 13, 13], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 576, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59bd7e71d22cd2700ef84808ff70e38f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_979512685801d46188aeb0d32fdcdda2
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_32089dd975aa5ae09d0e77a4bb9afc16(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d017977b054f5baef5251c9dda5194ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32089dd975aa5ae09d0e77a4bb9afc16
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.39149123430252075], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d017977b054f5baef5251c9dda5194ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32089dd975aa5ae09d0e77a4bb9afc16
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.39149123430252075], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f77c72d9dd9c5b058b46aab99ecf9960(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 8, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 8, 8], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8237ed23407431285d67112a2fa97507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f77c72d9dd9c5b058b46aab99ecf9960
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_96bdec893315e84077b46ee1d2a7eea7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 8, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3cd03d32297f9bd2f0d8dae66ab6ecd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96bdec893315e84077b46ee1d2a7eea7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3735694885253906], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7eba1aa160d29bbb44511b24b54850e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_52b195beb85374fc4492cc12d295237b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eba1aa160d29bbb44511b24b54850e0
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ac1a626eb1cf5bdff34976e8f96366d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 576, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 576, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d2083bd43ba2adec63b60ff510527681(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac1a626eb1cf5bdff34976e8f96366d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_91dc404a62a9ef53dc011df69dc0346f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 480, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_99166040128fa06a698c8e71efbc3d46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91dc404a62a9ef53dc011df69dc0346f
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_39db370e7e82a06db34491d3d41419ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5529, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5529, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23a080ca3a3f1437a90c1b9238610ab6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39db370e7e82a06db34491d3d41419ca
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23a080ca3a3f1437a90c1b9238610ab6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39db370e7e82a06db34491d3d41419ca
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23a080ca3a3f1437a90c1b9238610ab6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39db370e7e82a06db34491d3d41419ca
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23a080ca3a3f1437a90c1b9238610ab6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39db370e7e82a06db34491d3d41419ca
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23a080ca3a3f1437a90c1b9238610ab6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39db370e7e82a06db34491d3d41419ca
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_16a651dcd429f48194d9e53ec6b3a552(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5529, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[5529, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1cff2865d2f3847412790c27a41fa5d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16a651dcd429f48194d9e53ec6b3a552
    def get_inputs(self):
        return [
            paddle.uniform([5529, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1cff2865d2f3847412790c27a41fa5d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16a651dcd429f48194d9e53ec6b3a552
    def get_inputs(self):
        return [
            paddle.uniform([5529, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23a080ca3a3f1437a90c1b9238610ab6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39db370e7e82a06db34491d3d41419ca
    def get_inputs(self):
        return [
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5529, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ae8a89834b598d8d655b2b4378e3d301(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_81a90523a453f9594317ec6875d332bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae8a89834b598d8d655b2b4378e3d301
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6d9410094fe9637e3e6b73888ac26824(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 576, 15, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 576, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_56539153c693f0bcf12cedb32c83c5b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d9410094fe9637e3e6b73888ac26824
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_60bda26e4a9a5300d0796dce55d689a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 192, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2900d10b49b070ef86c58e2e81a6aa80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bda26e4a9a5300d0796dce55d689a4
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e1ad8cfdf8ad799f5c7266cc0e406353(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            paddle.static.InputSpec(shape=[36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6ee534ccdd0a7e660a73a4c065831dfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1ad8cfdf8ad799f5c7266cc0e406353
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ee534ccdd0a7e660a73a4c065831dfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1ad8cfdf8ad799f5c7266cc0e406353
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ee534ccdd0a7e660a73a4c065831dfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1ad8cfdf8ad799f5c7266cc0e406353
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aab31f782e3fa23dc3787266af269c40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 10, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe78224b6a0abe429acc8990311111fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aab31f782e3fa23dc3787266af269c40
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_77abb17d1cf32ac968d3cf6df91220de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70d4697ebff80b15d6b835649b35eb9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77abb17d1cf32ac968d3cf6df91220de
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4987477958202362], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_6082a6401640964013f8cd8561b907ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672, 17, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cbda0b58de8d6573cb865026be35a1a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6082a6401640964013f8cd8561b907ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9e8077b2266ae65fb8ba255331ccd0a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_77e583f4544510abc33ad8db6c875cf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e8077b2266ae65fb8ba255331ccd0a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f12c31ad5a0ed5599cd9083bee1205a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5eecf649e1f71ef4aa0c2c7b25dbe0d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f12c31ad5a0ed5599cd9083bee1205a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fe14fb69ea018dddceaaba5c0c8d4222(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 336, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 336, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b93131221efac9714a4b7e7d1312e988(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe14fb69ea018dddceaaba5c0c8d4222
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5985f9b73ca924c927275de1d8fd671c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 8, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4b2ee3d3d9324a824dca6ce684267077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5985f9b73ca924c927275de1d8fd671c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55a38fc41b063d0dd3b9da14c5a67689(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73ac9e1788bcb88b4f8ecb5ae2b2ae40
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ef10b1b63154d158c6d96593a58b1f5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 88, 88], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_83fb241d65837aa47f9834b6a0811ba9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef10b1b63154d158c6d96593a58b1f5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8c135422a2a2ad85191f982895874914(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15200, 81], dtype='float32'),
            paddle.static.InputSpec(shape=[15200, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_21c9ccb29019e96f326ed0b19a0ba943(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c135422a2a2ad85191f982895874914
    def get_inputs(self):
        return [
            paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_903b96e71dcc2edb4e7ec677c1e6f7d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 160, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1da8610a734a88af59753e98126840c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_903b96e71dcc2edb4e7ec677c1e6f7d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7f48b73f59d09160632ece643fcc9242(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 128, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_90b116a5114452afd399751c96a6464b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f48b73f59d09160632ece643fcc9242
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9751a1a0e3b426334234fe71bf0d4215(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 30, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_822bec2fe6b4b6df64d0d62b66c07ac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9751a1a0e3b426334234fe71bf0d4215
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a44734e17f450d9c0dbb1333fb739ac0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 10, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf8bd35711677dfbac1be7428223e1f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a44734e17f450d9c0dbb1333fb739ac0
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15870386362075806], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f7e8e9f2153719136b8e0a20b8cde799(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15200, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[15200, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6dadef67fd0493d474d3b0ca7245786c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7e8e9f2153719136b8e0a20b8cde799
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1053497eb037e1d471a225dd9de08bed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8949454426765442], dtype='float32').reshape([1]),
            paddle.to_tensor([0.43058085441589355], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b2619b4ae15cde5a2792832a721b5562(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.9048142433166504], dtype='float32').reshape([1]),
            paddle.to_tensor([0.26040756702423096], dtype='float32').reshape([1]),
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


class PrimitiveOp_0f4d4f7e433464980971678b4bc8860d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 144, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a1423ffef98080ff20503a0f41ac6707(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f4d4f7e433464980971678b4bc8860d
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_484bfd9d6d165443f303a2927a245e83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_62bb2e1f61e2ed7a0b0ea72287f3249c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_484bfd9d6d165443f303a2927a245e83
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8180e4b8ab9a80c3e815e033d0fdca0e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_46235f858dc80b7886b4461fa58db349(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8180e4b8ab9a80c3e815e033d0fdca0e
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_92cb17370c1018d4ec03807ee3b617dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 168, 168], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9eaf0f0ccadb2ee28f3b00261beec4a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92cb17370c1018d4ec03807ee3b617dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 168, 168], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_23d632587381744b828c29127c0434d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_82432e35387c083af9bbe5e617ccd42a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23d632587381744b828c29127c0434d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f50576135ce73f574a86b68017ca20fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 36, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 36, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_736560565c7f4dab4f47d7cffcfa87c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f50576135ce73f574a86b68017ca20fe
    def get_inputs(self):
        return [
            paddle.uniform([10, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_22aec529d4321b5699d224d6ffb96e74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_63c64d72e5caaad0d00c2973baa80d27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22aec529d4321b5699d224d6ffb96e74
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2787838578224182], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5db086effc395232daca9ffccee7db80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acb75ac7d8898f56f550b9ebe95dc4b7
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_743da1a40e2d519d599bc44cb5426535(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 12, 18], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cdf81c297a40c9e270a9a954b4d331e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743da1a40e2d519d599bc44cb5426535
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.35387247800827026], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1d2afc835e45d500c9b87ad1ecc06d22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca77bc397fbe63b61a2614e0c65c61fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d2afc835e45d500c9b87ad1ecc06d22
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2878d356e45aafddc05eef2c7e7a4c3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c9b691ab08eeb1c6a0c5324789037c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_06e3aeaa24fef692fbebc0f3db6bdcf8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 144, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57a22e9834dd91b2b43258cc81ba76ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06e3aeaa24fef692fbebc0f3db6bdcf8
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a362ef335a69f09b07b96404efdcbbd8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 672, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef94ad6ce939a558d0d894cebbb10130(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a362ef335a69f09b07b96404efdcbbd8
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            paddle.static.InputSpec(shape=[6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fa532f48bbaadfa44e7bce6806fcfb2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.17463308572769165, -0.2213263213634491, -0.3390448987483978, -0.08249804377555847, 0.0, 0.0007237792015075684], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.2035086750984192, -0.4491310715675354, -0.18829596042633057, -0.1561793088912964, -0.15362548828125, -0.32665085792541504], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_425b2e83746d15512192f4c3ff2dab5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03553934767842293, 0.09940452873706818, 0.06384078413248062, 0.012884487397968769, -0.0, -0.000236423104070127], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_59e6161f26bcd41285ce9970ad450e89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, -0.0, -0.000236423104070127], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_95d45631b6eca324430de71fcabc603a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3788352608680725, 0.11308163404464722, 0.2824884057044983, 0.2018137127161026, 0.0, 0.09728133678436279], dtype='float32').reshape([6]),
            paddle.to_tensor([0.02469462901353836, 0.0, 0.0, 0.0, 0.1565726101398468, 0.1054278314113617], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_31bb38e535157ba5d740cf68306bb8f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.17463308572769165, -0.2213263213634491, -0.3390448987483978, -0.017620444297790527, 0.368111252784729, 0.18772169947624207], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.006831318140029907, -0.05255473405122757, 0.2690444588661194, 0.09878036379814148, 0.10982172191143036, -0.16934990882873535], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_726c0f9fa8defe4413f659a8078e1058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10851454734802246, 0.08410894870758057, -0.13825735449790955, -0.1745946705341339, 0.07062837481498718, 0.14177775382995605], dtype='float32').reshape([6]),
            paddle.to_tensor([0.10851454734802246, 0.08410894870758057, -0.13825735449790955, -0.1745946705341339, 0.07062837481498718, 0.14177775382995605], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_c7f89dd49a28c85d6a8f682f11849b7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.2124403417110443, 0.23160921037197113, 0.29485806822776794, -0.2684791386127472, 0.286822646856308, 0.29468977451324463], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.2124403417110443, 0.23160921037197113, 0.29485806822776794, -0.2684791386127472, 0.286822646856308, 0.29468977451324463], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_696ab674e761085ffad11ecfcd0eeee8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3788352608680725, 0.11308163404464722, 0.2824884057044983, 0.26669132709503174, 0.368111252784729, 0.2842792570590973], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3788352608680725, 0.11308163404464722, 0.2824884057044983, 0.26669132709503174, 0.368111252784729, 0.2842792570590973], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_f52bbb4003d45dcb33923228c032b20e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22137197852134705, 0.3965763449668884, 0.45734041929244995, 0.25495967268943787, 0.42001980543136597, 0.262728750705719], dtype='float32').reshape([6]),
            paddle.to_tensor([0.22137197852134705, 0.3965763449668884, 0.45734041929244995, 0.25495967268943787, 0.42001980543136597, 0.262728750705719], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_78ccd50d8663e32d4621bcc353ca4442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.010535535402595997, 0.6586362719535828, 0.19265544414520264, 0.3391402065753937, 0.5352928042411804, -0.641175389289856], dtype='float32').reshape([6]),
            paddle.to_tensor([0.025995373725891113, 1.6251187324523926, 0.47535794973373413, 0.836794376373291, 1.3207812309265137, -1.582035779953003], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_3535686a939848d377b9ae7880ba08b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0002738002222031355, 0.5169926881790161, 0.08389698714017868, 0.22105678915977478, 0.4141785204410553, 0.5035650134086609], dtype='float32').reshape([6]),
            paddle.to_tensor([0.00027387519367039204, 1.0703620910644531, 0.09158029407262802, 0.28379061818122864, 0.7070046663284302, 1.0143624544143677], dtype='float32').reshape([6]),
        ]


class PrimitiveOp_a7bb475a7714f3d482ff50c213b4d533(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 480, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a3cc186b0340484e8c6312bdf79f567(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7bb475a7714f3d482ff50c213b4d533
    def get_inputs(self):
        return [
            paddle.uniform([10, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4512e33d194e9bbafd5ee55e70014114(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 60, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc43c2b9d3415f01dba026019590f702(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4512e33d194e9bbafd5ee55e70014114
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8f1f2ce33f99bc14e6cbe4b9b49b6cf3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 52, 52], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_71adc66b90be6e7589364be045596480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f1f2ce33f99bc14e6cbe4b9b49b6cf3
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_242556dee839e1cc8446a4f539a2711c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1767, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1767, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b083248c2920249f9999fb8e648e6cb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_242556dee839e1cc8446a4f539a2711c
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b083248c2920249f9999fb8e648e6cb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_242556dee839e1cc8446a4f539a2711c
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b083248c2920249f9999fb8e648e6cb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_242556dee839e1cc8446a4f539a2711c
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b083248c2920249f9999fb8e648e6cb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_242556dee839e1cc8446a4f539a2711c
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b083248c2920249f9999fb8e648e6cb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_242556dee839e1cc8446a4f539a2711c
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f69bf76bd6216a152dfd26aedd874833(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1767, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1767, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc8d23367be265ccb36f239d3784b9f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f69bf76bd6216a152dfd26aedd874833
    def get_inputs(self):
        return [
            paddle.uniform([1767, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc8d23367be265ccb36f239d3784b9f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f69bf76bd6216a152dfd26aedd874833
    def get_inputs(self):
        return [
            paddle.uniform([1767, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b083248c2920249f9999fb8e648e6cb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_242556dee839e1cc8446a4f539a2711c
    def get_inputs(self):
        return [
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1767, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99166040128fa06a698c8e71efbc3d46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91dc404a62a9ef53dc011df69dc0346f
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_46ed045afdd40a86ac71160f38bc97d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e591d5a8f5659df965e59db5ca98be49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46ed045afdd40a86ac71160f38bc97d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_beabe512e8cb4e2cc5f72f4d7797c01d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 256, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c52e73af637da6c3e3459b020b53c007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beabe512e8cb4e2cc5f72f4d7797c01d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_01d55c39be47665b8027112537f006d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 336, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 336, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e88d0ea2a78755fed68d071d047fd33a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01d55c39be47665b8027112537f006d1
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc3d59129f81ded7d193113e1dd756a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b20b0336e12a00d9214d414a53a93361
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_22a4d1373b2bac1f63849563cdaff296(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 1152, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 1152, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_323c304a201508ed6c471e016d5e34e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22a4d1373b2bac1f63849563cdaff296
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_155232ba38a5736633ca96c889848951(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 76, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f9f6e9dc7003a885d19ab2270b15586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_155232ba38a5736633ca96c889848951
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f74d0ba23ea63a3e855d3b191829cd80(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 92, 140], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ec89af43cfb36f9a171b03a695a6b9da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f74d0ba23ea63a3e855d3b191829cd80
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2127307504415512], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_458ab14b7d947a9d88ede4190462b605(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 32, 112, 112], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_76920a1da5a20aa34cfdca7151555f5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_458ab14b7d947a9d88ede4190462b605
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d31eee549a312ba6de8ed90c60c01200(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 20, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e92e5330a9bbede43db078e1910bdbe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d31eee549a312ba6de8ed90c60c01200
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f5aa15bd87331d1158da5b21ea7cae54(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2fccb049d664a968fe4d414fbe7c3f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5aa15bd87331d1158da5b21ea7cae54
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d310e75b7518c6e98e3259b0b624cbcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ba5f2f9d7280059f0268c44cc5db73b
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f458c247bbde0b501f5ffb146114f4ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 240, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[171, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_67946d551ebe9879442742c45f3ffd62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f458c247bbde0b501f5ffb146114f4ae
    def get_inputs(self):
        return [
            paddle.uniform([171, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9914dc33ff65efd9e6c755c3d3b201f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 336, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[171, 336, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c904ba1b14db15cedc86189c9bd2699c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9914dc33ff65efd9e6c755c3d3b201f5
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c1b34b55aabf7993d26444c37fd95b6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 480, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8dcf297576ad803742e643102e1a6289(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b34b55aabf7993d26444c37fd95b6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9869cb9d0710d9208427919ef767c7ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 38, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fd615772a6e6d58240d71888c908e8f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9869cb9d0710d9208427919ef767c7ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_baeca77b05250f7e43ddf14d51ae9565(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_48ed9765e2e14d290d3b48b0d748fa96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_baeca77b05250f7e43ddf14d51ae9565
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9a925e822f3f30e75c3ce588dc94246f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a0a2e651bbfded146b20c593a6cd6627(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a925e822f3f30e75c3ce588dc94246f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.29622042179107666], dtype='float32').reshape([1]),
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


class PrimitiveOp_49df513be1b8d3a4da5292a4568c65d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            paddle.static.InputSpec(shape=[24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a2556a188257a817bbee9134fda504b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49df513be1b8d3a4da5292a4568c65d1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.1924707889556885, 2.3366897106170654, 2.021646499633789, 2.194056272506714, 1.8130534887313843, 2.1789166927337646, 1.946984052658081, 2.0992658138275146, 2.1427483558654785, 1.9441338777542114, 1.9059215784072876, 2.15640926361084, 2.2924745082855225, 2.0655412673950195, 1.9126735925674438, 2.3783247470855713, 2.2228055000305176, 2.1653332710266113, 2.2527072429656982, 2.189174175262451, 2.1745641231536865, 1.9196083545684814, 1.9719960689544678, 2.0468955039978027], dtype='float32').reshape([24]),
            paddle.to_tensor([0.570573091506958, 0.7749462723731995, 0.887737512588501, 0.5992965698242188, 0.7103925347328186, 0.6079793572425842, 0.7386056184768677, 0.5993117690086365, 0.7632232904434204, 0.7227897644042969, 0.6032140851020813, 0.873702347278595, 0.9375947117805481, 0.5237308740615845, 0.944805920124054, 0.729169487953186, 0.604175329208374, 0.9376588463783264, 0.852842390537262, 0.7438666820526123, 0.5262297987937927, 0.8099640011787415, 0.9672985672950745, 0.6891385316848755], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_9e352584a4e337e61ee3af9f1713a5f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49df513be1b8d3a4da5292a4568c65d1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.9146963357925415, 2.0660901069641113, 1.8793545961380005, 1.8923225402832031, 2.1017744541168213, 2.030240297317505, 2.176668167114258, 1.9591494798660278, 1.9822040796279907, 2.06600284576416, 2.3405587673187256, 1.9868488311767578, 2.1357507705688477, 2.0293283462524414, 1.9782865047454834, 1.948639988899231, 1.9492480754852295, 2.212700843811035, 1.869426965713501, 2.014498233795166, 1.8967065811157227, 2.2487688064575195, 2.2292425632476807, 1.9269051551818848], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4294268786907196, 0.22505371272563934, 0.11226247251033783, 0.40070340037345886, 0.2896074652671814, 0.39202064275741577, 0.26139435172080994, 0.4006882309913635, 0.23677672445774078, 0.27721020579338074, 0.3967859148979187, 0.12629766762256622, 0.062405284494161606, 0.47626909613609314, 0.055194057524204254, 0.27083051204681396, 0.395824670791626, 0.06234115734696388, 0.14715760946273804, 0.2561332881450653, 0.4737702012062073, 0.19003598392009735, 0.03270141780376434, 0.3108614385128021], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_dac8f2e5323c0fc6bfb4db99c921da60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49df513be1b8d3a4da5292a4568c65d1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5182967185974121, 0.5689475536346436, 0.5014181137084961, 0.5182875990867615, 0.4741673171520233, 0.5301581025123596, 0.5017555356025696, 0.5107806921005249, 0.5261837840080261, 0.49447929859161377, 0.5195948481559753, 0.5337485074996948, 0.5706735253334045, 0.5120735168457031, 0.47907373309135437, 0.5654882192611694, 0.5286312103271484, 0.5420715808868408, 0.5490761399269104, 0.5361084342002869, 0.5107308626174927, 0.49554014205932617, 0.49510207772254944, 0.5023987889289856], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4493308365345001, 0.2919568121433258, 0.3668908476829529, 0.3289392590522766, 0.16020074486732483, 0.46689826250076294, 0.1551477015018463, 0.30581727623939514, 0.26257017254829407, 0.26085489988327026, 0.4375573396682739, 0.13953648507595062, 0.19184954464435577, 0.14506195485591888, 0.48614177107810974, 0.3297271430492401, 0.45540285110473633, 0.34352731704711914, 0.17210106551647186, 0.3333100378513336, 0.08750072121620178, 0.10533041507005692, 0.4893730580806732, 0.03919358551502228], dtype='float32').reshape([24]),
        ]


class PrimitiveOp_f3b4cc4ea02253ae05aa28eba571068f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 60, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[171, 60, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97d035dd98581d723918ca30a43b4bbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3b4cc4ea02253ae05aa28eba571068f
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


class TestPrimitiveOp_a5af04361edca3e1efb4d84b2d0d72e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c69efc00629a6fa23d714fdb6e006e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a5af04361edca3e1efb4d84b2d0d72e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c69efc00629a6fa23d714fdb6e006e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a5af04361edca3e1efb4d84b2d0d72e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c69efc00629a6fa23d714fdb6e006e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a5af04361edca3e1efb4d84b2d0d72e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c69efc00629a6fa23d714fdb6e006e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_653f25df76c1959f8c4ae7ccb5336e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c53df8392ce8625ae243f2ee7b563a5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_653f25df76c1959f8c4ae7ccb5336e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c53df8392ce8625ae243f2ee7b563a5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_653f25df76c1959f8c4ae7ccb5336e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c53df8392ce8625ae243f2ee7b563a5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d1af5cd1b6a317be079c9572b98c18da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1490, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1490, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0ff9a4badd6ce40d80fb1cacb3435321(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1af5cd1b6a317be079c9572b98c18da
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ff9a4badd6ce40d80fb1cacb3435321(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1af5cd1b6a317be079c9572b98c18da
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ff9a4badd6ce40d80fb1cacb3435321(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1af5cd1b6a317be079c9572b98c18da
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ff9a4badd6ce40d80fb1cacb3435321(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1af5cd1b6a317be079c9572b98c18da
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ff9a4badd6ce40d80fb1cacb3435321(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1af5cd1b6a317be079c9572b98c18da
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b1a4cdcfd8b2696b9b10bb9232cd7a98(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1490, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1490, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_45664551e2d2764777caace71c4c93eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1a4cdcfd8b2696b9b10bb9232cd7a98
    def get_inputs(self):
        return [
            paddle.uniform([1490, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45664551e2d2764777caace71c4c93eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1a4cdcfd8b2696b9b10bb9232cd7a98
    def get_inputs(self):
        return [
            paddle.uniform([1490, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ff9a4badd6ce40d80fb1cacb3435321(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1af5cd1b6a317be079c9572b98c18da
    def get_inputs(self):
        return [
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1490, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f925f5e691f99fb3d6b222aca7d1778a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 3549], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 3549], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e614bf359a2dc9dbbc652bdcc115d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f925f5e691f99fb3d6b222aca7d1778a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e614bf359a2dc9dbbc652bdcc115d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f925f5e691f99fb3d6b222aca7d1778a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_550fd531e9b0787d351dc130b2e59d13(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 3549], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_108a8506829c8a05a9329225fcb193ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_550fd531e9b0787d351dc130b2e59d13
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24917708337306976], [0.24660447239875793]]], dtype='float32').reshape([1, 2, 1]),
        ]


class PrimitiveOp_b83f557eacd8c4e2837e23f1dff73edf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3549, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ff16bb83648cace20b9c27d4c377a8dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b83f557eacd8c4e2837e23f1dff73edf
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


class PrimitiveOp_de528f7e6e64b9d54e1cde1c5384933d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 288, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 288, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6a6b75d871de3392b9ce3c332cdc9365(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de528f7e6e64b9d54e1cde1c5384933d
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_77d557ba6f6a303734664ea0cf392a52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32074b59bac5abaf1b8a9d10910ce304
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_77d557ba6f6a303734664ea0cf392a52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32074b59bac5abaf1b8a9d10910ce304
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_77d557ba6f6a303734664ea0cf392a52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32074b59bac5abaf1b8a9d10910ce304
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_77d557ba6f6a303734664ea0cf392a52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32074b59bac5abaf1b8a9d10910ce304
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_823e0a5156c49838cbf9d6d06d9c53b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 136, 136], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b83e7d0e05d84b196f91ab8baae5a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_823e0a5156c49838cbf9d6d06d9c53b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 136, 136], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_581a115768e3c7b13ce21c5b3054c571(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d760e43b8bcab16342b5849d4aa01b62
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c6063ee912be447da30887d64278c98a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 240, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_27d4e3812b177125ba3261172ab79fc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6063ee912be447da30887d64278c98a
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bd27cb57db362f26d74af4bb48557b9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3b14235645de248f3b5b931318ad47b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd27cb57db362f26d74af4bb48557b9b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.11154472827911377], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_2d4f4b41a1f8e98e927866cbdb496f4a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 76, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_50e48d6e40bd7c428eb4ee7e52e47506(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d4f4b41a1f8e98e927866cbdb496f4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e425d127966b14bc1ff56d6858f83625(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240, 20, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3513c8e0554d5f7e9c46ca26d702be7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e425d127966b14bc1ff56d6858f83625
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_90997f531f3da2136342d353b81fee34(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            paddle.static.InputSpec(shape=[4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d4bdb1f49d31cf56c59d1fb2aa9f3e6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90997f531f3da2136342d353b81fee34
    def get_inputs(self):
        return [
            paddle.to_tensor([2.1863529682159424, 1.9351545572280884, 2.1966986656188965, 2.0192975997924805], dtype='float32').reshape([4]),
            paddle.to_tensor([0.8553369045257568, 0.6661349534988403, 0.8746502995491028, 0.5330210328102112], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_7959d218d9698573eb178545622e81c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90997f531f3da2136342d353b81fee34
    def get_inputs(self):
        return [
            paddle.to_tensor([1.9532890319824219, 2.0163793563842773, 2.103386163711548, 2.1788883209228516], dtype='float32').reshape([4]),
            paddle.to_tensor([0.14466311037540436, 0.3338650166988373, 0.12534970045089722, 0.4669789671897888], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_545b3808a40cc34253a9e5371c3ad2a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90997f531f3da2136342d353b81fee34
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5381593108177185, 0.4905681610107422, 0.5462504625320435, 0.5234557390213013], dtype='float32').reshape([4]),
            paddle.to_tensor([0.0728219747543335, 0.026916885748505592, 0.06715060025453568, 0.4440712034702301], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_c293871a50f709ae77bd5ac416e7a68e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97eab3f01facdd16e7808d2e9a26d546
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c293871a50f709ae77bd5ac416e7a68e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97eab3f01facdd16e7808d2e9a26d546
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c293871a50f709ae77bd5ac416e7a68e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97eab3f01facdd16e7808d2e9a26d546
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c293871a50f709ae77bd5ac416e7a68e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97eab3f01facdd16e7808d2e9a26d546
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_326670e887bb48ce03d78c1dbd808ed3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 160, 44, 44], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d1befdbefd3dbf23b62ed6e7f8b3f90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_326670e887bb48ce03d78c1dbd808ed3
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1d7803907307b1ef52ed725b0700a107(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 80, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4aa4096adf5cda721a489afe5db768dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d7803907307b1ef52ed725b0700a107
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0e23867604478c3bdd9e77d3c65cfb38(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_85f2423af6bc94889d7d744110239ecf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e23867604478c3bdd9e77d3c65cfb38
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9e59d1345e52566a0de1041ad141a1c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2204, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[2204, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_73c3360d8a5438bc809dd2dc0ad2557a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e59d1345e52566a0de1041ad141a1c2
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef94ad6ce939a558d0d894cebbb10130(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a362ef335a69f09b07b96404efdcbbd8
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ba81e29fe6eac68f9bfbc2813a1c4bb2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 36, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 36, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e487a152c56f3d9efda2ef80071884e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba81e29fe6eac68f9bfbc2813a1c4bb2
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0399b6b5890bd2e28b6c4105967fdbad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc8d8d4e371b030accea017e68e231ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0399b6b5890bd2e28b6c4105967fdbad
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1773347407579422], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e0b9f03b1162d46c3768e32b17f3c7dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_14651988a2d79cd45d9cfcb2131fcc82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0b9f03b1162d46c3768e32b17f3c7dd
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_41ad33f09467481b34e88fb829ddf1b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0b9f03b1162d46c3768e32b17f3c7dd
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3096567988395691]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0892241969704628]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_744da234d3d1dd995f5a9081d63d15f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0b9f03b1162d46c3768e32b17f3c7dd
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.006796002388000488]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.1392716020345688]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_9dfd10f6ba687f0c83cb0cef8bd51a6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0b9f03b1162d46c3768e32b17f3c7dd
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3096567988395691]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.28117379546165466]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_f806701357b7f72ee71db5f39236ad4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6f4f2695ad16c7c77e69861404b436
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1b2780f69624fff122ed8c64c70bbb6d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f5af1a06b0be91da909be27a8b48f7d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b2780f69624fff122ed8c64c70bbb6d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.1069101095199585], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.014185458421707153], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_b812c266b0c27390a477aa611a44165c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b2780f69624fff122ed8c64c70bbb6d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.27551791071891785], [0.029222995042800903], [0.1069101095199585], [0.23753564059734344], [-0.20119929313659668], [-0.014492303133010864]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.03136730194091797], [0.11883988976478577], [0.18468628823757172], [0.004471004009246826], [-0.21776650846004486], [-0.33405452966690063]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_6184f87b697f184f8006f62b61212160(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b2780f69624fff122ed8c64c70bbb6d
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.056698232889175415], [-0.0009203255176544189], [0.3283199667930603], [-0.23824363946914673], [0.06682560592889786], [0.013432860374450684]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.0708579421043396], [-0.24443936347961426], [0.15392956137657166], [0.14829422533512115], [-0.007320474833250046], [-0.0210229754447937]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_94792fdb315633145579ea0a436938a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b2780f69624fff122ed8c64c70bbb6d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.27789831161499023], [0.06557220220565796], [0.3283199667930603], [0.2462882697582245], [0.06682560592889786], [0.10599508881568909]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.007773280143737793], [0.11883988976478577], [0.3244304060935974], [0.15295250713825226], [0.149659663438797], [0.00729137659072876]], dtype='float32').reshape([6, 1]),
        ]


class PrimitiveOp_83ab3565ebf3df8809d393520af0ed08(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ad2ee10ab1be7a1659236b617cfca40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83ab3565ebf3df8809d393520af0ed08
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6eb0e15a01a84c60f193e0d07796a34b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0733dba543fb438834e2dc9ba0f387bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6eb0e15a01a84c60f193e0d07796a34b
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a64fa566b317e48c6abbc2182f3c730e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 144, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b34aa6d684b559c6abdb5d9e0d72fd3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a64fa566b317e48c6abbc2182f3c730e
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0a456f0aa8c14d027aa36c5d72198f1a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 24, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a0502cc894d72fdb4a4d550249ef634(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a456f0aa8c14d027aa36c5d72198f1a
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class PrimitiveOp_781ca45ad191072a9fda64884f7b50ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 104, 104], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e9e4ab18760bcd945054ac5e22f73c24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_781ca45ad191072a9fda64884f7b50ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_43a769bc28dbce6a70906be4d6c49877(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 184, 184], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b675ee52ab2602c736d81efdf66ba4f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43a769bc28dbce6a70906be4d6c49877
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 184], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b79dd7060ce3ed8be9a67757b148a4d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 128, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c34954440c726710b60dd42fa40341c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b79dd7060ce3ed8be9a67757b148a4d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.7921505570411682]], [[0.740813672542572]], [[0.7466079592704773]], [[0.8749993443489075]], [[0.8499155044555664]], [[0.8249661922454834]], [[0.8377474546432495]], [[0.9854068756103516]], [[0.8102701902389526]], [[0.8981375694274902]], [[0.8567355871200562]], [[0.8062554597854614]], [[0.8787116408348083]], [[0.8138719201087952]], [[0.7610911130905151]], [[0.766205906867981]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class PrimitiveOp_425de543daf868ba0cce0b31447a73ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 52, 52], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ca88a948f25931fad6fbb02267c60fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_425de543daf868ba0cce0b31447a73ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cd6884c7e30465b66269ab96919ba6bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a5559c326348a46ce6c09f518a6a1ace(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd6884c7e30465b66269ab96919ba6bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f42814ead87e7fa1a2c611bd00bbd84c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 15, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf7479c69d8df217aefb815c694e20c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f42814ead87e7fa1a2c611bd00bbd84c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5aed20c7e958de824a596996be23cd1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 60, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[145, 60, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f44185fe16d75d897679b34bf07036e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aed20c7e958de824a596996be23cd1b
    def get_inputs(self):
        return [
            paddle.uniform([145, 60, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bd3206ed2fe7ac43381c2bbb0e5d2f9c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 10, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4f70897830a0dc0e2952bc96c9de039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd3206ed2fe7ac43381c2bbb0e5d2f9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4488915801048279], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_df7011d3bb16432e13116eb1e12c4c50(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[70, 81], dtype='float32'),
            paddle.static.InputSpec(shape=[70, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ad64d67089060d675140c0f986f22b04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df7011d3bb16432e13116eb1e12c4c50
    def get_inputs(self):
        return [
            paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d0c7be1670664e593a2faa44e7eedefd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672, 32, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a9e75c6802082d4da77eec4a09a2d6e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0c7be1670664e593a2faa44e7eedefd
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_bedfb4cad42f393c4a752b799ed23a16(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 4116], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 4116], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3999825377318f379238b1687d5126fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bedfb4cad42f393c4a752b799ed23a16
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3999825377318f379238b1687d5126fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bedfb4cad42f393c4a752b799ed23a16
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a5ea2b5f2f1308e8175fb854b8b66069(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 4116], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_38d5f9beac148bedad8fc8978e4a9b8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5ea2b5f2f1308e8175fb854b8b66069
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24771934747695923]]], dtype='float32').reshape([1, 1, 1]),
        ]


class PrimitiveOp_e7bebefd0aea2c8a9b29a927a116b0b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 4116, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cdf8cd152a5cea8bbba6d9be96804115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7bebefd0aea2c8a9b29a927a116b0b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_67dc2be8bf074d228354b5ae558bbe37(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[551, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[551, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a227c89f7e1a7506fd0e03f95dc0d925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67dc2be8bf074d228354b5ae558bbe37
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6c6280990710f6cdd77bc4f0ea46c442(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 400, 22, 22], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 400, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0fe9195a524015f4940a0ed9771e0236(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c6280990710f6cdd77bc4f0ea46c442
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d15fe55ef3097276ca57da60c4d5cba8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 336, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 336, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1222b669b5aa45a3d343d0c1e2ddfa0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d15fe55ef3097276ca57da60c4d5cba8
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_911be43c55a1c2e24ab4cb563f65e439(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 240, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d13f6e9b10039d6ac1219852840327e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_911be43c55a1c2e24ab4cb563f65e439
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_17973d10c572415966e8a183af596682(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240, 18, 18], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b0f77e202b352b0b5f787cdf737f9c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17973d10c572415966e8a183af596682
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab592917f0852e524a311a54046e3084(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626ae5005bb74d0d19e833cd0b4aec1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab592917f0852e524a311a54046e3084(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626ae5005bb74d0d19e833cd0b4aec1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab592917f0852e524a311a54046e3084(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626ae5005bb74d0d19e833cd0b4aec1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab592917f0852e524a311a54046e3084(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626ae5005bb74d0d19e833cd0b4aec1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e294f50de410d34a91184777073d8595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2f5611870d3f1d8af810feb55e461d
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3a42a2723e1e5d7cfc7f59b0c8067d4b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 9, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c2d8357f0a7d09e4b03f96b6502d1618(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a42a2723e1e5d7cfc7f59b0c8067d4b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0f9f741b9eec58ea6d971c878a6899c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_398fa9c09478fe3daee7b79939ee8174(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f741b9eec58ea6d971c878a6899c5
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b5e21e9893a0ee7929b613cf12ea4de8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[247, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[247, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_653afb7b6a0ba0c829ed56aaec49dad9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5e21e9893a0ee7929b613cf12ea4de8
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_329d627d1789960dbc7ad0c6299c5604(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672, 20, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6147453ef4b88fb2ea6a3028491248a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_329d627d1789960dbc7ad0c6299c5604
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2b4cad3a8e65763df84c2b3237594a16(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 88, 88], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5dda7aa5dc6b9d116f62c10f99c3435b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b4cad3a8e65763df84c2b3237594a16
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_45337722230d20739d122b436c01825f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 336, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 336, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_adc5cb57b0afefeed7c29bf5d3e897a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45337722230d20739d122b436c01825f
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_91c445d21139e34473e8f53ae00a1e56(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 104, 104], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b4fb33f7341bcd4684990e1426d02b2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91c445d21139e34473e8f53ae00a1e56
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 104, 104], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a100a88fd07868b4e4af1eb096d6600f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[950, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[950, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f0f1eeccaf44d8e43acb711b85ca46a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a100a88fd07868b4e4af1eb096d6600f
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5a6ac6e417aa2f9e70b7543858833ed7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 32, 112, 112], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_32f8710be3693ace0c84c80f151525b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a6ac6e417aa2f9e70b7543858833ed7
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


class PrimitiveOp_ea0be2fd07a5b31350bc3ce58eceaa31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 144, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d11668bd4dde9cc6df78951b7cfc53d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea0be2fd07a5b31350bc3ce58eceaa31
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1bdb49101bf446c654cc08ec506e2a3e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_85ce723ff3d663e34076aaf839665d2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bdb49101bf446c654cc08ec506e2a3e
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58a379cc4e133b750eab41f9ab80825c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b4af2e724b4b2010f490551ae44b0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58a379cc4e133b750eab41f9ab80825c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b4af2e724b4b2010f490551ae44b0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58a379cc4e133b750eab41f9ab80825c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b4af2e724b4b2010f490551ae44b0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_88266f0a88e56578e8a70fa0f3a56d8c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 52, 52], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b7ac3174e07b84ca3b9fe55fde1c6f23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88266f0a88e56578e8a70fa0f3a56d8c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7eb265886210cdcdffd9bb7c69061d80(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 44, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 44, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c9f982a694b2916eaf06a1d64a184442(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eb265886210cdcdffd9bb7c69061d80
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_13831282a9b56e6317826d2bfd8f8fb6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bf19100b110b6803fd9e2e71e554555c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13831282a9b56e6317826d2bfd8f8fb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c57e5b50d1f9ddf8c6a107db66b6f238(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 400, 13, 13], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 400, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b5c7201b61d991e79dc95ddb4032123(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c57e5b50d1f9ddf8c6a107db66b6f238
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d835d8df3c2e970a3a833e74afa70962(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 56, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a19294942a098fcbe2b29e7edcca747f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d835d8df3c2e970a3a833e74afa70962
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.47053638100624084], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_eee649416a3696b12166505bb1d12aa7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[950], dtype='float32'),
            paddle.static.InputSpec(shape=[950], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_71c0a3fbf40f7282edb7b15cd30aaac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eee649416a3696b12166505bb1d12aa7
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_67448f2337e9c3d3bcfd73b04524295f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240, 128, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f93cc350e5976a2f04f9c2ed056a1198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67448f2337e9c3d3bcfd73b04524295f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_711d61c2ab9ee86e44eb33a3f879f802(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 56, 60, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 56, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e3c6c68c61afa442b20f386ed2e8c612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711d61c2ab9ee86e44eb33a3f879f802
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0c49b4c9d4c069b5e0893f4249f7b850(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8816], dtype='float32'),
            paddle.static.InputSpec(shape=[8816], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b654e03fc2859e2cc7f978dc2f0c0d51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c49b4c9d4c069b5e0893f4249f7b850
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_81c6476a079cae352d720238d3560c97(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2010, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2010, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66e2ac357bee976d843599a9fabc3ca8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81c6476a079cae352d720238d3560c97
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66e2ac357bee976d843599a9fabc3ca8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81c6476a079cae352d720238d3560c97
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66e2ac357bee976d843599a9fabc3ca8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81c6476a079cae352d720238d3560c97
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66e2ac357bee976d843599a9fabc3ca8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81c6476a079cae352d720238d3560c97
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66e2ac357bee976d843599a9fabc3ca8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81c6476a079cae352d720238d3560c97
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_db07e39060b3a52d05f8acf1217f0ecb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2010, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2010, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b7069f1c3bbdb576223e4d4cc5aa312d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db07e39060b3a52d05f8acf1217f0ecb
    def get_inputs(self):
        return [
            paddle.uniform([2010, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b7069f1c3bbdb576223e4d4cc5aa312d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db07e39060b3a52d05f8acf1217f0ecb
    def get_inputs(self):
        return [
            paddle.uniform([2010, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66e2ac357bee976d843599a9fabc3ca8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81c6476a079cae352d720238d3560c97
    def get_inputs(self):
        return [
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2010, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0250f32c341932761eafea94816444b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 15, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_905c77dd64f1c6b4ebee95d223bda517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0250f32c341932761eafea94816444b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72cfe43a08fb9b696196feef4292a72a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7234470844268799], dtype='float32').reshape([1]),
            paddle.to_tensor([0.02489776536822319], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fede03ec3ab59a5f4122245f6dcabe62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7354514002799988], dtype='float32').reshape([1]),
            paddle.to_tensor([0.06607259809970856], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_387339e995d9340d5c978fa4019f57a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.9486072659492493], dtype='float32').reshape([1]),
            paddle.to_tensor([0.4159499406814575], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_89e5d025dfce8b3c90a1f3ddf9265930(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.6703647375106812], dtype='float32').reshape([1]),
            paddle.to_tensor([0.17629185318946838], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_674609d3903eb125b51a8f296c11e318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8651716709136963], dtype='float32').reshape([1]),
            paddle.to_tensor([0.2052282840013504], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bc1d0c67350ec6ee2a346d28ea18008e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8071532845497131], dtype='float32').reshape([1]),
            paddle.to_tensor([0.030724747106432915], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9a375701fd31580e463ad78158a0751b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8010026812553406], dtype='float32').reshape([1]),
            paddle.to_tensor([0.2503751814365387], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fe97221c45e474fa5d8b98230d16384f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8924924731254578], dtype='float32').reshape([1]),
            paddle.to_tensor([0.32399311661720276], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8e094b437782e574ef9fc2b8a461d483(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8923561573028564], dtype='float32').reshape([1]),
            paddle.to_tensor([0.3656964898109436], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f98ea5d32539ef9702904a3797f06e1e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8816, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[8816, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee7de8242fbb3632e76b87e5f56d1321(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f98ea5d32539ef9702904a3797f06e1e
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_094c190833ee8d62c1961fdea27f1fae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e4afc22c3644ae0e613ba56a2e9f66e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_094c190833ee8d62c1961fdea27f1fae
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e4afc22c3644ae0e613ba56a2e9f66e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_094c190833ee8d62c1961fdea27f1fae
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e4afc22c3644ae0e613ba56a2e9f66e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_094c190833ee8d62c1961fdea27f1fae
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e4afc22c3644ae0e613ba56a2e9f66e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_094c190833ee8d62c1961fdea27f1fae
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0dda6823b552830379006659d462860d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16384, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[16384, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_68a6f2bd71cc3aaff7d37090b34a0755(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0dda6823b552830379006659d462860d
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_513ae1739a28cafbf5128434a2ca254b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 42, 42], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_31a1fa8bf60eabd225917f7ac6c838e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_513ae1739a28cafbf5128434a2ca254b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 42, 42], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2d1ee8bbdd03eebeb42367a09f3b6c7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 60, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39f61135e161b308bf2ea46278049d85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d1ee8bbdd03eebeb42367a09f3b6c7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_90412ca909e2194393aa35bb44fdba24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ff296a469b7a09f083e3882bdb499a8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90412ca909e2194393aa35bb44fdba24
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3387312591075897], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ff296a469b7a09f083e3882bdb499a8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90412ca909e2194393aa35bb44fdba24
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3387312591075897], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_aefdace9d19840fd2c02dda73359eba4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f11e7aaf9b69d16af86a57763a4968d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aefdace9d19840fd2c02dda73359eba4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8d7f35f039e7ad2049eb45bad5e4b525(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_377c7709e1b9af14e68ecec3d3c44ec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d7f35f039e7ad2049eb45bad5e4b525
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14177025854587555], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_429f77e7f7302cf79fc882d07e53131d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 160, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4fe63008963914bd57053586152a3293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_429f77e7f7302cf79fc882d07e53131d
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aa3b319a493f507b319a5ef0d7b4cc68(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 30, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_666b2f94bed83bd963d654f02f646d98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa3b319a493f507b319a5ef0d7b4cc68
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8c55ad37c141f2018c41856cd0bb634a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4663, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4663, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1fb7adfe773326ab134a70eaa08d42b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c55ad37c141f2018c41856cd0bb634a
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1fb7adfe773326ab134a70eaa08d42b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c55ad37c141f2018c41856cd0bb634a
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1fb7adfe773326ab134a70eaa08d42b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c55ad37c141f2018c41856cd0bb634a
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1fb7adfe773326ab134a70eaa08d42b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c55ad37c141f2018c41856cd0bb634a
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1fb7adfe773326ab134a70eaa08d42b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c55ad37c141f2018c41856cd0bb634a
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fa23440cac82af4309abb1a5becd56d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4663, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[4663, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_11a537fd63f70eb245090133b976bb82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa23440cac82af4309abb1a5becd56d5
    def get_inputs(self):
        return [
            paddle.uniform([4663, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11a537fd63f70eb245090133b976bb82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa23440cac82af4309abb1a5becd56d5
    def get_inputs(self):
        return [
            paddle.uniform([4663, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1fb7adfe773326ab134a70eaa08d42b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c55ad37c141f2018c41856cd0bb634a
    def get_inputs(self):
        return [
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4663, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1423ffef98080ff20503a0f41ac6707(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f4d4f7e433464980971678b4bc8860d
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_84ffd7f472527bec19f44199ce5cac64(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4acd4f5dfe0e1f1919ca35a77ecfa07b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84ffd7f472527bec19f44199ce5cac64
    def get_inputs(self):
        return [
            paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.18413560092449188]], [[0.10664638131856918]], [[0.39406290650367737]], [[0.336351603269577]], [[0.036685071885585785]], [[0.13355301320552826]]], dtype='float32').reshape([6, 1, 1]),
        ]


class PrimitiveOp_7ada4132836aa661971ac9d154b79896(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 19, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_98c69f1f1a65f8178e3847097d7f52a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ada4132836aa661971ac9d154b79896
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_aeb659cb9f3548098381dedb4f4ba628(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1090, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1090, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b3ed09d422343f7dfc6b32c7e81fb65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aeb659cb9f3548098381dedb4f4ba628
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b3ed09d422343f7dfc6b32c7e81fb65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aeb659cb9f3548098381dedb4f4ba628
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b3ed09d422343f7dfc6b32c7e81fb65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aeb659cb9f3548098381dedb4f4ba628
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b3ed09d422343f7dfc6b32c7e81fb65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aeb659cb9f3548098381dedb4f4ba628
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b3ed09d422343f7dfc6b32c7e81fb65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aeb659cb9f3548098381dedb4f4ba628
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2dee423aaa79cce33d822f2b1acb2527(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1090, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1090, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1aaab80489d0b6c0472e6c752f3549bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2dee423aaa79cce33d822f2b1acb2527
    def get_inputs(self):
        return [
            paddle.uniform([1090, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1aaab80489d0b6c0472e6c752f3549bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2dee423aaa79cce33d822f2b1acb2527
    def get_inputs(self):
        return [
            paddle.uniform([1090, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b3ed09d422343f7dfc6b32c7e81fb65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aeb659cb9f3548098381dedb4f4ba628
    def get_inputs(self):
        return [
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1090, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f80d0ab208ce3eed5377f86b6cbe16fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0bcedd0441c713524a19cc4eb3f1d3ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f80d0ab208ce3eed5377f86b6cbe16fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1d6d9eb946c5ddbd8b2a7896ee540eec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 96, 30, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[2, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a536cafcac8e6f8bb7989c4c641e29a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d6d9eb946c5ddbd8b2a7896ee540eec
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_934c25cb1f28f1284c4a62c7a4fddccf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 96, 60, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[2, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9727e3f7654330131bea0e1e104a9bf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_934c25cb1f28f1284c4a62c7a4fddccf
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_564aa4804d2ab58f4f8afc374ce66739(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 96, 120, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[2, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fa960b8ee44ea7ab4b8468702e7dcd67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_564aa4804d2ab58f4f8afc374ce66739
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_dcedc90d2624fedf3a4962d18492559f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 96, 240, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[2, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1c04b591be5937f2e8f268c5fab7e430(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcedc90d2624fedf3a4962d18492559f
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2c7f864af7eaaeefb609476efe0becd5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 24, 30, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[2, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a5892a3722c105a042ca425361e18abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c7f864af7eaaeefb609476efe0becd5
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_691d6d737f0d01cb0a668e4c3cfb1df0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 24, 60, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[2, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2d227304bef8a59dcde9b574817d511b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_691d6d737f0d01cb0a668e4c3cfb1df0
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7ffe346411e55f42fdce51b4b4d9aeeb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 24, 120, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[2, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a075165494681a9817732a44c3089311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ffe346411e55f42fdce51b4b4d9aeeb
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_18473e19dd47ed997b04632d88ab08e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 24, 240, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[2, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_840b769722487a9c07d00a6dbeedc8b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18473e19dd47ed997b04632d88ab08e8
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_323c304a201508ed6c471e016d5e34e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22a4d1373b2bac1f63849563cdaff296
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9c8b58b96a0112d6b478b87a960a84ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 20, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_66d5e6b3b78809e24ee21b18322040ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c8b58b96a0112d6b478b87a960a84ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b34aa6d684b559c6abdb5d9e0d72fd3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a64fa566b317e48c6abbc2182f3c730e
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_be4d5f661cb938009ef15686ba9dbd5f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 8, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ad0f49105b6b637091ea1ed9f00689f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be4d5f661cb938009ef15686ba9dbd5f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fd5bc996e3ff67991d0ac104d81f369b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_19b1d4e254dcf10eeaeaf422e9aaea21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd5bc996e3ff67991d0ac104d81f369b
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44d7f426003eaeb0bfc01911508afb62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b24ad60e80ab24bb6b066b48623adab2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44d7f426003eaeb0bfc01911508afb62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b24ad60e80ab24bb6b066b48623adab2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44d7f426003eaeb0bfc01911508afb62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b24ad60e80ab24bb6b066b48623adab2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6ec6cd942b64db2c8fcf7d19a07cf177(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_30a03f03c8752fdf219b3fa2e711ee78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ec6cd942b64db2c8fcf7d19a07cf177
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3d8d81b646705ca1ac48d7efa39592c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[100, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_18c5af37efd9950b75831a948c871629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d8d81b646705ca1ac48d7efa39592c8
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_18c5af37efd9950b75831a948c871629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d8d81b646705ca1ac48d7efa39592c8
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2d431583dbce2806d5aea6fac77553e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 44, 66], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_329c46febbd4a03a40cc95ddc4414b56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d431583dbce2806d5aea6fac77553e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.110385961830616], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_50a61ad11597cfa1516622801b49ecda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 20, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e83959cfab17542fb0f2535714be3a6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50a61ad11597cfa1516622801b49ecda
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e528cf0e2f60be445b356baeceb7805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed1ba3c28e87e79667627287243f3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e528cf0e2f60be445b356baeceb7805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed1ba3c28e87e79667627287243f3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e528cf0e2f60be445b356baeceb7805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed1ba3c28e87e79667627287243f3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e528cf0e2f60be445b356baeceb7805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed1ba3c28e87e79667627287243f3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fcdb0534aaf89be4838b4682250534cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 200, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 200, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6e8d54cb0932fb081a17d74a86f0325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fcdb0534aaf89be4838b4682250534cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a68a05971af9c854fa27a115447b4d5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_88f23ff519986e6fa1b4916bf9259c39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a68a05971af9c854fa27a115447b4d5c
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


class TestPrimitiveOp_81a90523a453f9594317ec6875d332bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae8a89834b598d8d655b2b4378e3d301
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_81a90523a453f9594317ec6875d332bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae8a89834b598d8d655b2b4378e3d301
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_81a90523a453f9594317ec6875d332bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae8a89834b598d8d655b2b4378e3d301
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7224d3bbe2685b29d75fb20bdf4e3e8f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2048, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00bd04d3e952f54a7a6fe6d6b8e5285c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7224d3bbe2685b29d75fb20bdf4e3e8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_48c1f0b3aa5a6b888dacff0000fcac2d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_04d7a3919e4961157399a355d0fc96a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48c1f0b3aa5a6b888dacff0000fcac2d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9d2c9a26cef048330e50504fc2591152(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[300, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80e9c4da21a6a2dcc372e1a751982797(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d2c9a26cef048330e50504fc2591152
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_80e9c4da21a6a2dcc372e1a751982797(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d2c9a26cef048330e50504fc2591152
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_71914385028c60457327f20b50af0ff7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f59c3ceb8b66f82e3f86f7a016baa379(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71914385028c60457327f20b50af0ff7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.15678969025611877], [0.0], [0.000620424747467041], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0855332612991333]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_436c5d6842ff050d7cc796a7d439c232(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71914385028c60457327f20b50af0ff7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.012531213462352753], [0.2674189507961273], [0.010834276676177979], [0.24491703510284424], [0.3781909942626953]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.11443698406219482], [-0.06015931814908981], [-0.0034473538398742676], [-0.053752630949020386], [0.38662946224212646]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_f0f8b2d2fd65980065d33aed28e639d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71914385028c60457327f20b50af0ff7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06411457061767578], [0.2883461117744446], [-0.02012452483177185], [0.000620424747467041], [-0.10309046506881714]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.09969541430473328], [-0.19413119554519653], [-0.3575252890586853], [-0.08874985575675964], [0.11003163456916809]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_c3aac6ad3676b42c368a13579fe1532d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71914385028c60457327f20b50af0ff7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3533153533935547], [0.3989753723144531], [0.056439489126205444], [0.24491703510284424], [0.3781909942626953]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.025193482637405396], [0.08111090958118439], [-0.0034473538398742676], [0.042302489280700684], [0.41112783551216125]], dtype='float32').reshape([5, 1]),
        ]


class PrimitiveOp_020f692312cb5a1a1f46a6a93b1b8895(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 68, 68], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_93aa80e458e80604bd2b5263568e79a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_020f692312cb5a1a1f46a6a93b1b8895
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b035ba371ec230244f260b4bb69367b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57123c3c0cfae45806d46ba4493338e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b035ba371ec230244f260b4bb69367b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a3be3c6893175a59793a356157187c79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1755a3b55313704e9d532054e970f0d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3be3c6893175a59793a356157187c79
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_97eb22375d38f1d6c87856e02c2f16c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4245913426c50e23e26e996c4e3e55d4
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cbbe62f8db8441233de6fbcdd5240f62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d07132c8b6e809630ba97c8e007c0d37
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1095d3fb2ffa7ebd7c82d25b853d012c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2df90e7fe7ffdc7f4eb66c6e3d30b961
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


class PrimitiveOp_bdbb53a9fcb2ba0cbe12e542d72858bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 76, 116], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e71739783e87d1105553356270a3192c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdbb53a9fcb2ba0cbe12e542d72858bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 76, 116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.19148683547973633], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d6a76a7eda1110b5ac220b554f1120a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 100, 44, 44], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 100, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c24618db431136ffb52423c4c2b06914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6a76a7eda1110b5ac220b554f1120a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f93cc350e5976a2f04f9c2ed056a1198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67448f2337e9c3d3bcfd73b04524295f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9902438db4ef593149b654f30dd9b70b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 288, 10, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 288, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a5cda640a36a2fc2d712eb6a9cb0949(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9902438db4ef593149b654f30dd9b70b
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b036567a61f9ed516a526e013a1b778c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e0e5c2552ce620b675021cd9f097a45d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b036567a61f9ed516a526e013a1b778c
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_515ba5bae4ea344adfb82b718c674e68(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f8f6a5fc7d6c2cd0748d35c23f183c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_515ba5bae4ea344adfb82b718c674e68
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_08cb7f26b76ddd9a4c61c34b99be7b82(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e33d286afe83d515f1a19225150d8cd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08cb7f26b76ddd9a4c61c34b99be7b82
    def get_inputs(self):
        return [
            paddle.to_tensor([0.04596744105219841], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c2dc37b7073e12ea906b00baa546f793(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_860be1dd550a879e8438049a9424451b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2dc37b7073e12ea906b00baa546f793
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_860be1dd550a879e8438049a9424451b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2dc37b7073e12ea906b00baa546f793
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_860be1dd550a879e8438049a9424451b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2dc37b7073e12ea906b00baa546f793
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fbc0814c6bd68ab928c1368e48637c25(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 2048, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ba4069ca8b5aed2ad3f5ca95bbce4fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbc0814c6bd68ab928c1368e48637c25
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8fcbc8e940e1477629be8048e2b3c30b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1248, 10, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1248, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_51a27fd0c488774348a74e4fcdbd1a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fcbc8e940e1477629be8048e2b3c30b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1248, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5637e340af3332d7784201c0c63c067c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 288, 30, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 288, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d98734be2c6c6a6331fa43ba772d1435(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5637e340af3332d7784201c0c63c067c
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5fcf03a6086130ae9486abb2f881899c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 128, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c64bcb7a1e6d600c72a00861cb5e877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fcf03a6086130ae9486abb2f881899c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.09809057414531708], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4c64bcb7a1e6d600c72a00861cb5e877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fcf03a6086130ae9486abb2f881899c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.09809057414531708], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_afd8fc5e9722ed38aa8e16a950fe160b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 128, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 128, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c7dc93a9ed4fddc1e89af320cfad7e26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afd8fc5e9722ed38aa8e16a950fe160b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e955f135d79f9cab1a2d7bece733a436(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 128, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ea2ff953c46b819faa1bc5e3a04c0889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e955f135d79f9cab1a2d7bece733a436
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.10970340669155121], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6cb7e69001d0e18af310bc3823478ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83c3cdb6e829f210bfd695bbc50ad272
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cb7e69001d0e18af310bc3823478ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83c3cdb6e829f210bfd695bbc50ad272
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6cb7e69001d0e18af310bc3823478ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83c3cdb6e829f210bfd695bbc50ad272
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_57d04499dbe9fd9080bb3e8010b53874(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 480, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[171, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_757a44699a5ba4713be7b90b3ee631d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57d04499dbe9fd9080bb3e8010b53874
    def get_inputs(self):
        return [
            paddle.uniform([171, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([171, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11e808ae9b13b846a14edbfa88601359(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d636a839c1554b6e0420f4abb46f545f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11e808ae9b13b846a14edbfa88601359(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d636a839c1554b6e0420f4abb46f545f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11e808ae9b13b846a14edbfa88601359(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d636a839c1554b6e0420f4abb46f545f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_60bf61eef8fe289dfd9d842a35208591(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_83ed84ee28d0bf9012dcc39b6b00b606(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bf61eef8fe289dfd9d842a35208591
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83ed84ee28d0bf9012dcc39b6b00b606(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bf61eef8fe289dfd9d842a35208591
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83ed84ee28d0bf9012dcc39b6b00b606(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bf61eef8fe289dfd9d842a35208591
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8f36b1c24ce4555f892835091fa7d24d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 36, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[145, 36, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_efb2af467e81e6b05467cc1a5258c75b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f36b1c24ce4555f892835091fa7d24d
    def get_inputs(self):
        return [
            paddle.uniform([145, 36, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55a38fc41b063d0dd3b9da14c5a67689(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73ac9e1788bcb88b4f8ecb5ae2b2ae40
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7138806c53fe65047584c00ade942317(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 5, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_688ce079ee17447f2185e066ce3acd36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7138806c53fe65047584c00ade942317
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.19824975728988647], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9c0fe7ab31e463748f80f95818a23727(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 256, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d8344dce1b7b823acd9045d7ab2d0f51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c0fe7ab31e463748f80f95818a23727
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_107d3bbe2a770bb1bd427861fb29d15d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 52, 52], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_912e63aa7d9e6e13e5ec532feb791eb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_107d3bbe2a770bb1bd427861fb29d15d
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_03175c544330f9d37e4e8afbbf3f1ca2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 23, 23], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_17743355c50d5c77a0f0d57d1790f463(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03175c544330f9d37e4e8afbbf3f1ca2
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 23, 23], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2be875a9688084abb7cd42c3f26dfeab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2374, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2374, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6c4dd7c737288022f109934770e4d776(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2be875a9688084abb7cd42c3f26dfeab
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c4dd7c737288022f109934770e4d776(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2be875a9688084abb7cd42c3f26dfeab
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c4dd7c737288022f109934770e4d776(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2be875a9688084abb7cd42c3f26dfeab
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c4dd7c737288022f109934770e4d776(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2be875a9688084abb7cd42c3f26dfeab
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c4dd7c737288022f109934770e4d776(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2be875a9688084abb7cd42c3f26dfeab
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_34fce12a8ec8e2dd4692fd49499228a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2374, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2374, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_36d4b85dd7236036da4e4edd7da8dfd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34fce12a8ec8e2dd4692fd49499228a6
    def get_inputs(self):
        return [
            paddle.uniform([2374, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36d4b85dd7236036da4e4edd7da8dfd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34fce12a8ec8e2dd4692fd49499228a6
    def get_inputs(self):
        return [
            paddle.uniform([2374, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c4dd7c737288022f109934770e4d776(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2be875a9688084abb7cd42c3f26dfeab
    def get_inputs(self):
        return [
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2374, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_efec768d4ce4f13c30448ea8fb310dbe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ce4904c843da0fd9d2a03ae5de0bd242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efec768d4ce4f13c30448ea8fb310dbe
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1ff51b96b9086fc098740f35a266b6e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 240, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_85ce0d5f9bce96cf3efe344fc5e7a486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ff51b96b9086fc098740f35a266b6e9
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2a4e4ded013e0556ef1d504866342cbd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3058, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3058, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b411f6480276d58d58d4a4848c8cb687(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a4e4ded013e0556ef1d504866342cbd
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b411f6480276d58d58d4a4848c8cb687(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a4e4ded013e0556ef1d504866342cbd
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b411f6480276d58d58d4a4848c8cb687(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a4e4ded013e0556ef1d504866342cbd
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b411f6480276d58d58d4a4848c8cb687(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a4e4ded013e0556ef1d504866342cbd
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b411f6480276d58d58d4a4848c8cb687(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a4e4ded013e0556ef1d504866342cbd
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e4e276aaac04f2b3d25a904fe80d12c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3058, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[3058, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_67a5f82eb98473a17927df2e253938e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4e276aaac04f2b3d25a904fe80d12c7
    def get_inputs(self):
        return [
            paddle.uniform([3058, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67a5f82eb98473a17927df2e253938e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4e276aaac04f2b3d25a904fe80d12c7
    def get_inputs(self):
        return [
            paddle.uniform([3058, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b411f6480276d58d58d4a4848c8cb687(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a4e4ded013e0556ef1d504866342cbd
    def get_inputs(self):
        return [
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3058, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b3d7250126b6b1879540b817825ec3f0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3793, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3793, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d9fa7f5b29e20d0a5e5b312391d7eb23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3d7250126b6b1879540b817825ec3f0
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d9fa7f5b29e20d0a5e5b312391d7eb23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3d7250126b6b1879540b817825ec3f0
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d9fa7f5b29e20d0a5e5b312391d7eb23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3d7250126b6b1879540b817825ec3f0
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d9fa7f5b29e20d0a5e5b312391d7eb23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3d7250126b6b1879540b817825ec3f0
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d9fa7f5b29e20d0a5e5b312391d7eb23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3d7250126b6b1879540b817825ec3f0
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c888e08db218704310bf2244b49ef3db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3793, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[3793, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c4dd4e261d699116ac41793fe30a8f52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c888e08db218704310bf2244b49ef3db
    def get_inputs(self):
        return [
            paddle.uniform([3793, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4dd4e261d699116ac41793fe30a8f52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c888e08db218704310bf2244b49ef3db
    def get_inputs(self):
        return [
            paddle.uniform([3793, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d9fa7f5b29e20d0a5e5b312391d7eb23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3d7250126b6b1879540b817825ec3f0
    def get_inputs(self):
        return [
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3793, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f904b4249c94af8f5bd043b44f301a7a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ccf38a389841b2387ae93a368b6f679a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f904b4249c94af8f5bd043b44f301a7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3056052625179291], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ccf38a389841b2387ae93a368b6f679a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f904b4249c94af8f5bd043b44f301a7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3056052625179291], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_260c21a37e265b18f9213c152646cf98(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f7843c211acf82a7b5fec378e2ff6f38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_260c21a37e265b18f9213c152646cf98
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a20ef40aebeb098fea9c3756d8eee753(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d19fecb0a09d2756ab9b185d565aa67d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a20ef40aebeb098fea9c3756d8eee753
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.19838325679302216], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_7b07049406cfd624569bc2827f1dac0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 156, 40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 156, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_90854fed9a2899dd568c37399e9eec49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b07049406cfd624569bc2827f1dac0b
    def get_inputs(self):
        return [
            paddle.uniform([1, 156, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 156, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4e3021d5ccb2b4db6189e868f531c7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06afa0ac8681719902f3e86d0feb1b9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.9517311453819275]], [[0.9340465664863586]], [[0.9168874621391296]], [[0.9462327361106873]], [[0.9227688908576965]], [[0.9301791787147522]], [[0.8818508386611938]], [[0.9246761202812195]], [[0.8667358160018921]], [[0.8598313927650452]], [[0.949042022228241]], [[0.8845770955085754]], [[0.9282195568084717]], [[0.9245629906654358]], [[0.9301644563674927]], [[0.9204380512237549]], [[0.8777945041656494]], [[0.8192024230957031]], [[0.854398787021637]], [[0.8927462697029114]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


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


class PrimitiveOp_502a7eb46deb36385e27cdfcc2c1fa37(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 92, 92], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4fb0fdd033e74a342cfad60817e46969(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_502a7eb46deb36385e27cdfcc2c1fa37
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 92, 92], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0f3a704bbef93b0703c705ca2a8835d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256, 5], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 5], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de2ec90864af150f6afcf3317d6a0673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f3a704bbef93b0703c705ca2a8835d2
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1c7bdc80da0b2006252a28c70adb41a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 20, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ece16a5d1ec4490d5c0d0bb901dac299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c7bdc80da0b2006252a28c70adb41a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_22af0ad1c13eed434c8b50308b1d8241(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 200, 44, 44], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 200, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e366ec01384335e573e7bea47e20f7cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22af0ad1c13eed434c8b50308b1d8241
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_77edbdf76945be9cef813f9fcc8152a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 52, 52], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_712bb94572d989adbecd2247d885c33e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77edbdf76945be9cef813f9fcc8152a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f1e5c99c02e707f89895497c2deb623a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 80, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_259c63bf4c67e75fa0059b0705efa11c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1e5c99c02e707f89895497c2deb623a
    def get_inputs(self):
        return [
            paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class PrimitiveOp_9a349844fda90726045bdef2e5166888(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 320, 9, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b9d799e203fc16eb9d6e853b0c63d3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a349844fda90726045bdef2e5166888
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e4afc22c3644ae0e613ba56a2e9f66e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_094c190833ee8d62c1961fdea27f1fae
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e4afc22c3644ae0e613ba56a2e9f66e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_094c190833ee8d62c1961fdea27f1fae
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e4afc22c3644ae0e613ba56a2e9f66e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_094c190833ee8d62c1961fdea27f1fae
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_112fd303e9afcabd718ff6300e652eff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08cb7f26b76ddd9a4c61c34b99be7b82
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12507888674736023], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5acca4d79884cb836a8cf2ff2b46102f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9eea7664498eaa2d2fd5ce326f4049e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5acca4d79884cb836a8cf2ff2b46102f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3202affc3aba5367ae9a98124a1547b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 96, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23dfaffd59348d629ae42d5dfff7c608(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3202affc3aba5367ae9a98124a1547b7
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0ea64f73170f3a50b33c2c158d5e402d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 34, 34], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4f68db5d9a3927aface81c67a7da2575(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ea64f73170f3a50b33c2c158d5e402d
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_860a59f28792c56fa166071671e35112(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 672, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a62428a1c4af547d1be9dcce9ff70bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_860a59f28792c56fa166071671e35112
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0bcedd0441c713524a19cc4eb3f1d3ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f80d0ab208ce3eed5377f86b6cbe16fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7606825c99fce1b673f6c40ad63754af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 872, 10, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 872, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c86196f11666cee385b1544299f074aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7606825c99fce1b673f6c40ad63754af
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d066935a2d690803217a9d50515ac25a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[247], dtype='float32'),
            paddle.static.InputSpec(shape=[247], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f47749016904c798b6de639269587fab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d066935a2d690803217a9d50515ac25a
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a5e7aa5d68a57e6474e9d1052ed0fe1e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dbc429126a91c7531ad968184bd1c843(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e7aa5d68a57e6474e9d1052ed0fe1e
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


class PrimitiveOp_c713bdacca1358dc085fa03dedacad1d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 480, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39eec619a32487962d92752498105129(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c713bdacca1358dc085fa03dedacad1d
    def get_inputs(self):
        return [
            paddle.uniform([22, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0616a5ceabe962aa3b2a8edd4642fa70(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 40, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3950fa4ed65a7e8cf498130231a019e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0616a5ceabe962aa3b2a8edd4642fa70
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_89e3fd03c0dad8dcdb4883f123d86317(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[145, 480, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[145, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7cae4789b41f3e251071de0cf63f8643(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89e3fd03c0dad8dcdb4883f123d86317
    def get_inputs(self):
        return [
            paddle.uniform([145, 480, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([145, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cf95189c0e8c270fd0d242cc8a5f9adc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 40, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_14aac92d3e6c06c069099e4ac4b1d510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf95189c0e8c270fd0d242cc8a5f9adc
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.08327114582061768], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_5edaa4c79b377a22877dbe9049949df9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[2, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f25a63b30be84fcafe7527d4cbc4ec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5edaa4c79b377a22877dbe9049949df9
    def get_inputs(self):
        return [
            paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.16857312619686127]], [[0.09711317718029022]]], dtype='float32').reshape([2, 1, 1]),
        ]


class PrimitiveOp_ad9b0a72a1a93921ab0c11808ad69640(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 46, 46], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b1a64ee4a6fab7751f955ebe0d062e02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad9b0a72a1a93921ab0c11808ad69640
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 46, 46], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0c7b980800ab890863c28f9a24ff437a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[171, 36, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[171, 36, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc1c11b8e55e3b392e147eab02be3853(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c7b980800ab890863c28f9a24ff437a
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


class PrimitiveOp_2b6836b520a449ffacec29e71582bff5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 10, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3eb4d91a1bf341701cefa34494f1deb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b6836b520a449ffacec29e71582bff5
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_00ee895733957654bcc26beabf3a9205(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_259227e7b20438bb1ff59dbe9765998b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00ee895733957654bcc26beabf3a9205
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c8537dfae7567de2fa5575c59c20debe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 68, 68], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_614815675bd7da61ab4b642215635937(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8537dfae7567de2fa5575c59c20debe
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_396325b24b038419a11b6092c7a134be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_65b3399827bb1bc089eb3e2dfd00f3fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396325b24b038419a11b6092c7a134be
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f773f57b350be988ac85f55a5df29df3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 288, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 288, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c8b59ecf662ee467d9c9c1677c81e432(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f773f57b350be988ac85f55a5df29df3
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


class PrimitiveOp_5d1fa69c7bf246f8ce1d40d1fc06910e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc833837540eab4ed33caefaf9b25452(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d1fa69c7bf246f8ce1d40d1fc06910e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b860cca7d74e41392aecd7a90ee59238(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            paddle.static.InputSpec(shape=[20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_63854637fdac2eb3ab51846ddd6462f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b860cca7d74e41392aecd7a90ee59238
    def get_inputs(self):
        return [
            paddle.to_tensor([1.9683747291564941, 2.084948778152466, 2.2118701934814453, 2.1216206550598145, 2.1139209270477295, 2.190563440322876, 2.0532028675079346, 2.1269404888153076, 1.8814973831176758, 2.2099437713623047, 2.278904438018799, 2.280040740966797, 2.1386146545410156, 2.1280288696289062, 2.0714845657348633, 2.0619211196899414, 2.242865562438965, 1.986429214477539, 2.282512903213501, 2.337796211242676], dtype='float32').reshape([20]),
            paddle.to_tensor([0.9752313494682312, 0.7573302984237671, 0.9274436235427856, 0.7018563747406006, 0.9366440176963806, 0.9862791895866394, 0.7997195720672607, 0.9336206912994385, 0.7224514484405518, 0.7648215889930725, 0.5857998132705688, 0.8997608423233032, 0.8205776214599609, 0.782891571521759, 0.682951807975769, 0.7092351913452148, 0.751106858253479, 0.6151508092880249, 0.6111956834793091, 0.6527165174484253], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_a5e62a01d8a5c45d50924ea1cf685eec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b860cca7d74e41392aecd7a90ee59238
    def get_inputs(self):
        return [
            paddle.to_tensor([1.9860702753067017, 2.2777862548828125, 2.1725013256073, 2.3833882808685303, 2.043961763381958, 2.150498867034912, 2.1957919597625732, 2.049513578414917, 2.186617136001587, 2.082493305206299, 1.9323686361312866, 2.274631977081299, 2.2910077571868896, 1.8456910848617554, 2.042461395263672, 1.9607930183410645, 2.2835118770599365, 2.154902458190918, 2.1080117225646973, 1.9610350131988525], dtype='float32').reshape([20]),
            paddle.to_tensor([0.024768635630607605, 0.2426697313785553, 0.07255639135837555, 0.2981436252593994, 0.06335600465536118, 0.013720804825425148, 0.20028044283390045, 0.06637927889823914, 0.27754852175712585, 0.23517842590808868, 0.41420018672943115, 0.10023913532495499, 0.17942240834236145, 0.21710842847824097, 0.31704822182655334, 0.29076477885246277, 0.2488931566476822, 0.3848491609096527, 0.3888043463230133, 0.3472835123538971], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_49cfcf02b5ca55929d9e968b0d8c72ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b860cca7d74e41392aecd7a90ee59238
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4922032356262207, 0.532936155796051, 0.5522534251213074, 0.5499162673950195, 0.5273721814155579, 0.5475034713745117, 0.5204401612281799, 0.5304502248764038, 0.491545706987381, 0.5449925661087036, 0.5338423252105713, 0.5698745846748352, 0.5414893627166748, 0.5166827440261841, 0.5155707001686096, 0.5081291198730469, 0.5632455348968506, 0.5128164887428284, 0.5536665320396423, 0.5517383217811584], dtype='float32').reshape([20]),
            paddle.to_tensor([0.07258226722478867, 0.3994954526424408, 0.21663706004619598, 0.3020555377006531, 0.028603216633200645, 0.03494539484381676, 0.22202180325984955, 0.02695070020854473, 0.42817097902297974, 0.4212222695350647, 0.20056572556495667, 0.3698383569717407, 0.012171325273811817, 0.16965487599372864, 0.25900083780288696, 0.40566012263298035, 0.28427526354789734, 0.39007046818733215, 0.33132487535476685, 0.3903793692588806], dtype='float32').reshape([20]),
        ]


class PrimitiveOp_786d583c57b98a30e25d9f6524f2c827(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672, 10, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_77af636fd1ede98a671b128921a7aabb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_786d583c57b98a30e25d9f6524f2c827
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90b116a5114452afd399751c96a6464b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f48b73f59d09160632ece643fcc9242
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e3e2ef532b1395ffc405f5306b0727a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672, 38, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e7dfb50425f5e26d865bfd97cb87ae5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3e2ef532b1395ffc405f5306b0727a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_15d40efaf2971c8ae33fea5f84371db1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[247, 81], dtype='float32'),
            paddle.static.InputSpec(shape=[247, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6429d8dbbe336dba903f1e5cfa3dab88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15d40efaf2971c8ae33fea5f84371db1
    def get_inputs(self):
        return [
            paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_752a8f3f78ec45b5d0d71fb92718baa5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672, 16, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_022ace9072a002cdbaa2600a20e1a6e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752a8f3f78ec45b5d0d71fb92718baa5
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f5b4acd9abd9ffb99bb81463fec102cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240, 84, 84], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6592d56742b8dc934b692d2007eaa25f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5b4acd9abd9ffb99bb81463fec102cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 84, 84], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_db8033965d9a3d28afc98b44727dd707(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a03debe7205bcc11946cc0a0e8f98b4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db8033965d9a3d28afc98b44727dd707
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_81026698d78d85743c53d72b67e75e59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06afa0ac8681719902f3e86d0feb1b9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.8999562859535217]], [[0.8337211012840271]], [[0.8533642292022705]], [[0.7548016905784607]], [[0.9264510869979858]], [[0.7980631589889526]], [[0.8523966073989868]], [[0.7939204573631287]], [[0.8417139649391174]], [[0.9319472312927246]], [[0.8978875875473022]], [[0.9035201072692871]], [[0.9227541089057922]], [[0.8426836133003235]], [[0.8515221476554871]], [[0.88536137342453]], [[0.9131758809089661]], [[0.9374407529830933]], [[0.8221920132637024]], [[0.8824988007545471]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_20a228bd881e6abe9c2150c1edcb899f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08a35fac70f021394b01b6711933f04f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b98eb3e500bcacfd0b31f99ca2396226(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34967eb0bb98058b9f8a9a8cc0d2f7e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23dfaffd59348d629ae42d5dfff7c608(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3202affc3aba5367ae9a98124a1547b7
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


class PrimitiveOp_740ad0093b4ec2fa38d08f27bbe1086f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 160, 36, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6f3550508c027bfd093535da12aa5480(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740ad0093b4ec2fa38d08f27bbe1086f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b03ffb5ee15dafa65de3266a43786ab2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_373f28edd813211622ed1dd23f15b0b4
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7a62428a1c4af547d1be9dcce9ff70bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_860a59f28792c56fa166071671e35112
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_548273f689e30af802afc00c29cff5d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 80, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eb2a445108b83a95c1d29a198ba0e4f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_548273f689e30af802afc00c29cff5d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d11668bd4dde9cc6df78951b7cfc53d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea0be2fd07a5b31350bc3ce58eceaa31
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3adb20594a6eb3588fd3da022747d609(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 48, 48], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9461773a4840a3c786b21489371c577c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3adb20594a6eb3588fd3da022747d609
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_73356518337e2f5c65479b43248a9b8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0c9ff29f5d9dd67407e0f168bb337f4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73356518337e2f5c65479b43248a9b8e
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


class PrimitiveOp_eb69369f90a7508e0efc5e3cddd2dbd1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_855744111e8823c8c3b40459705a472a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb69369f90a7508e0efc5e3cddd2dbd1
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_da68c0f6e208c8f3ca8c5df3de35c420(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 336, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 336, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d45f685470e70802e1926897e8e53859(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da68c0f6e208c8f3ca8c5df3de35c420
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2080653d0e0c3645fa313a40c82a396b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 128, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b5d19207ab7b7c20aa5ba8e1b8564286(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2080653d0e0c3645fa313a40c82a396b
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_82aedc620f8680bcf4a39a1c8efd8eb0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 12, 12], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_095f8981079ea9f4dd045f786a0f18b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82aedc620f8680bcf4a39a1c8efd8eb0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_99399950c34280c4c605c0aadc4c35f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 56, 48, 48], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 56, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f2f5fd26ff581638791c40fbcdb269cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99399950c34280c4c605c0aadc4c35f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_509dc620ff46007c0b8a2e4224bed677(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[950, 81], dtype='float32'),
            paddle.static.InputSpec(shape=[950, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f633e0744a07ae822edb2db29beb1f88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_509dc620ff46007c0b8a2e4224bed677
    def get_inputs(self):
        return [
            paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
            paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f714bbf021d42dc1978a94f2e6726786(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 576, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 576, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_36cbac0a5fccb960c3291a3bb40e8d2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f714bbf021d42dc1978a94f2e6726786
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a2d80b0a464dd4db39b3fdc7fe062c41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1787b8f155c3f2615ba954c1567a596e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2d80b0a464dd4db39b3fdc7fe062c41
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.1181102991104126], [0.0], [0.08827203512191772]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_e6428685eb7e0439c61d3c92028ba96d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2d80b0a464dd4db39b3fdc7fe062c41
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.048685409128665924], [0.4554782807826996], [-0.1235523670911789], [0.4803711175918579]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.19099071621894836], [0.12709885835647583], [0.10437566041946411], [-0.20124077796936035]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_56a89c0d0651f203a666a3906fa23d37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2d80b0a464dd4db39b3fdc7fe062c41
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.14594703912734985], [0.1181102991104126], [-0.2242138832807541], [0.08827203512191772]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.3008948564529419], [-0.002876684069633484], [-0.09710609912872314], [0.013299329206347466]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_8772ee5db87962f175e69f8d8fd2e64f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2d80b0a464dd4db39b3fdc7fe062c41
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1287224292755127], [0.4554782807826996], [-0.1235523670911789], [0.4803711175918579]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.19099071621894836], [0.42696940898895264], [0.10437566041946411], [0.049181193113327026]], dtype='float32').reshape([4, 1]),
        ]


class PrimitiveOp_dc3ec5b0206724b7e1f435e15a92fd19(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 36, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fd605d58f64005c24924635581ced118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc3ec5b0206724b7e1f435e15a92fd19
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8bc4720b1c97bc36613c2f8acca09784(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            paddle.static.InputSpec(shape=[70], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_56342d0e94813c15a7e0b34ec398aa5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8bc4720b1c97bc36613c2f8acca09784
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8dcf297576ad803742e643102e1a6289(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b34b55aabf7993d26444c37fd95b6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ef01a48e23abc3212b0acf8572ff9e26(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 320, 15, 15], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3d30ddb68c390981abc4909e3b9b416e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef01a48e23abc3212b0acf8572ff9e26
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36cbac0a5fccb960c3291a3bb40e8d2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f714bbf021d42dc1978a94f2e6726786
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_538500537929b6b1c466453ae69ded23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 76, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bd610859409d178171306d71efdc922f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_538500537929b6b1c466453ae69ded23
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6a36ba04b41c7c843bedbb4d423b2a09(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 40, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ea5a94806e675ca1dcf51473b3684fb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a36ba04b41c7c843bedbb4d423b2a09
    def get_inputs(self):
        return [
            paddle.uniform([43, 40, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1a1b927eb195547e01b6d7d96990efc6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 19, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_368b88c9aeb83d092d35d7dcd2aa73d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a1b927eb195547e01b6d7d96990efc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1ccd931db1b27da5d28537e49231823f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6aee25dd7e46be8e50e618df0e3b05f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ccd931db1b27da5d28537e49231823f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f3ad5d76ed43394f1063c579701c619b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 160, 18, 18], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_091ad17d81147f0cb8e00db615ff0547(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3ad5d76ed43394f1063c579701c619b
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b1ee2cd06d1d24d22551b85b6397a2ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2042, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2042, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a9da4197fe50bc38e0ad32bafe0d370(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1ee2cd06d1d24d22551b85b6397a2ae
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a9da4197fe50bc38e0ad32bafe0d370(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1ee2cd06d1d24d22551b85b6397a2ae
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a9da4197fe50bc38e0ad32bafe0d370(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1ee2cd06d1d24d22551b85b6397a2ae
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a9da4197fe50bc38e0ad32bafe0d370(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1ee2cd06d1d24d22551b85b6397a2ae
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a9da4197fe50bc38e0ad32bafe0d370(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1ee2cd06d1d24d22551b85b6397a2ae
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_580e98ebd22918087918844ca99ec62e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2042, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2042, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d4243a34b19115f05c05eb532913ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_580e98ebd22918087918844ca99ec62e
    def get_inputs(self):
        return [
            paddle.uniform([2042, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d4243a34b19115f05c05eb532913ccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_580e98ebd22918087918844ca99ec62e
    def get_inputs(self):
        return [
            paddle.uniform([2042, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a9da4197fe50bc38e0ad32bafe0d370(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1ee2cd06d1d24d22551b85b6397a2ae
    def get_inputs(self):
        return [
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2042, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_448cf51c0a1fffe56b88bcff73ae7eda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 20, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_858d78929f235a228b5629f62bc58214(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_448cf51c0a1fffe56b88bcff73ae7eda
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3b5bfa76040afbc7d0b8f79979650f5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 120, 120], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f3361bd4f7432457782d2f9430273cad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b5bfa76040afbc7d0b8f79979650f5c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e08d7f1689a76970e8a0fa8557059a8f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 21, 21], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_50e5ea15281571b4e86eea60bf8a2302(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e08d7f1689a76970e8a0fa8557059a8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 21, 21], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_497700affc2049fe9771b15ab19815f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 97, 97], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_906a767e2bfa7e87fddf0930fc8e52e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_497700affc2049fe9771b15ab19815f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f4282a9406db450d904526d6461a1776(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 38, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc3895e7b4b890c31906d36d0afbd82a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4282a9406db450d904526d6461a1776
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ec9ee9d1e6d1914b48ea8d88287b037c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[70, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[70, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0106ab02e35c034fd591a720f2c897f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec9ee9d1e6d1914b48ea8d88287b037c
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_815e72764af3389ec9d9aaca8c653179(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea3ad6a4af220e8de872eaa59cebe711
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1841116db056a60f14d7b14423cde2a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0399b6b5890bd2e28b6c4105967fdbad
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2057395875453949], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_002eb7d3088a9f42b9dd7b94a65a4e24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 256, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef509efe6c614eceb1f1d9ae9822b87a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002eb7d3088a9f42b9dd7b94a65a4e24
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


class PrimitiveOp_e5aad34d7c86157d1487993f25703db5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e4cadd7be796b05a981a6c002b6e2a43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5aad34d7c86157d1487993f25703db5
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4158576726913452], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e4cadd7be796b05a981a6c002b6e2a43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5aad34d7c86157d1487993f25703db5
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4158576726913452], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_8573bc190e2cdda526bce2f19a07883c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, 32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ddf9073c0d77ceee29ef155e6ef5e414(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8573bc190e2cdda526bce2f19a07883c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_7e2e2d1970fd88b18fd9d1b4d7b374ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_05fc5af262f27ba576b43ab6a1474a9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e2e2d1970fd88b18fd9d1b4d7b374ba
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.36707356572151184], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_d36208301922b0d255a3a1653ba98568(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 200, 18, 18], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 200, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a949a8a07cfe410fda6fa7e4a33f3d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d36208301922b0d255a3a1653ba98568
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4a3d3de657dfd343ed2575898fb6f779(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 400, 9, 9], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 400, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8234e2253decb6678cbca772e988171c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a3d3de657dfd343ed2575898fb6f779
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e3683b728cb38e102bc836e28eeeea7a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 68, 68], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fecc800f72a4c0f2ff89289c30c20ea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3683b728cb38e102bc836e28eeeea7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_321399b2c2bf46627156294e15157a86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[551], dtype='float32'),
            paddle.static.InputSpec(shape=[551], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bca7dfca3710a5bcc2c916907c75bf6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_321399b2c2bf46627156294e15157a86
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32f8710be3693ace0c84c80f151525b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a6ac6e417aa2f9e70b7543858833ed7
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ac7b51980a8ac159818613a8b9043cc3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d613bab1b34b621a6f7014d4b8b791bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac7b51980a8ac159818613a8b9043cc3
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83ed84ee28d0bf9012dcc39b6b00b606(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bf61eef8fe289dfd9d842a35208591
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83ed84ee28d0bf9012dcc39b6b00b606(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bf61eef8fe289dfd9d842a35208591
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83ed84ee28d0bf9012dcc39b6b00b606(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bf61eef8fe289dfd9d842a35208591
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83ed84ee28d0bf9012dcc39b6b00b606(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bf61eef8fe289dfd9d842a35208591
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


class PrimitiveOp_99762ac386920a1faa3d3da1d7e6c1df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 100, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 100, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4fb23a05ec52cd4b8aff9e77a5daf619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99762ac386920a1faa3d3da1d7e6c1df
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_749e7ebb28b45c42341bfa042c093e44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5e00ddb5250321479259e986cdd2528
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4aa673b6f0d385e88c4482bb417d5b5f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 48, 48], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a5908e8dfc03bed8b398cd4bbb20c162(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4aa673b6f0d385e88c4482bb417d5b5f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8b59ecf662ee467d9c9c1677c81e432(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f773f57b350be988ac85f55a5df29df3
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_27041bd2d04e95bf6250eed180c2bd1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3800], dtype='float32'),
            paddle.static.InputSpec(shape=[3800], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1f826295970c3e68815b0b758421f20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27041bd2d04e95bf6250eed180c2bd1c
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76920a1da5a20aa34cfdca7151555f5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_458ab14b7d947a9d88ede4190462b605
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_26032e168cdc9c2cca0df1b3c85ec3a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8098960eb115a79955652f745c57d4f2
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_4ff4032347afccea751b9d7652d07ac0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 160, 88, 88], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_03ab017ae943c205cecdc2296f7edd22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ff4032347afccea751b9d7652d07ac0
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e7f241038e30db9e6912d0782751485c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 320, 13, 13], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dda1ef8daaf70c93f4a2a7eea2337445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7f241038e30db9e6912d0782751485c
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_860be1dd550a879e8438049a9424451b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2dc37b7073e12ea906b00baa546f793
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cd79497871969a862d4fbceea36d0efa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 48, 48], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_35e70dd4a75d6fe86924c3f1f7a15a33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd79497871969a862d4fbceea36d0efa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b3a875073a11ba5ab64d9b4598a017dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 38, 58], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_950608e4dc4b36d449a21f203bb37068(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3a875073a11ba5ab64d9b4598a017dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 38, 58], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.38453152775764465], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4d13f6e9b10039d6ac1219852840327e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_911be43c55a1c2e24ab4cb563f65e439
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_76285b4e18b0326c79be4007407e9750(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2204], dtype='float32'),
            paddle.static.InputSpec(shape=[2204], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c28440519c82b58249ca8b0c7828d682(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76285b4e18b0326c79be4007407e9750
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d651b3382091c8333b6aeb3076a83d6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 112, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1d592b4221fd01abf2ed5525f217f449(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d651b3382091c8333b6aeb3076a83d6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3729340434074402], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_432c37706dfe1d909f11741a7efb46a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 160, 52, 52], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b3325c849e37adbda08ca60f863abffe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_432c37706dfe1d909f11741a7efb46a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1ab8a4edf80a48e3437fcca6a1eb9987(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cb98750e9222179cc274e84e1236fb10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ab8a4edf80a48e3437fcca6a1eb9987
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2fccb049d664a968fe4d414fbe7c3f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5aa15bd87331d1158da5b21ea7cae54
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c85f2f8d2852001006ac18e16d09892e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53bf962e1cdee93256bf89ec3da751ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c85f2f8d2852001006ac18e16d09892e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.015576010569930077], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_85ce0d5f9bce96cf3efe344fc5e7a486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ff51b96b9086fc098740f35a266b6e9
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_67c444cd82fc1906f1b4ead121f105db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 256, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ea5660c501f2e3f16a56f263de6b3ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67c444cd82fc1906f1b4ead121f105db
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db48a6cc79be5577a4cb6273e7aba3e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8253ad6111de39b8930be7b782affcb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db48a6cc79be5577a4cb6273e7aba3e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8253ad6111de39b8930be7b782affcb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db48a6cc79be5577a4cb6273e7aba3e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8253ad6111de39b8930be7b782affcb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db48a6cc79be5577a4cb6273e7aba3e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8253ad6111de39b8930be7b782affcb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5d19207ab7b7c20aa5ba8e1b8564286(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2080653d0e0c3645fa313a40c82a396b
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e5cc0de66dde8037696c9f8a4333633d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 23, 41], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eb33ae6c4bde824e0cdad7e41838e4ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5cc0de66dde8037696c9f8a4333633d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_126a8025592276f3c231b9564272106f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 46, 82], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab3207afb248fff094f5a7eed4b31ee5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_126a8025592276f3c231b9564272106f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cc6aac554478e161f1adc661708a0055(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 92, 164], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_76d25a2cdaec932840583b8ea86c612b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc6aac554478e161f1adc661708a0055
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e7179123bbda5856709d935e6dd10d96(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 184, 328], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ddd7ff0b134ade09a12d4c7074b6c2bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7179123bbda5856709d935e6dd10d96
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a4b73c091f5ae1b12cd82ee0aff7fda4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 24, 23, 41], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0f3fae88380f420fa7ce68d917551d16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4b73c091f5ae1b12cd82ee0aff7fda4
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 23, 41], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class PrimitiveOp_15f7a64633c9f07e2e70f08658436418(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 24, 46, 82], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ad42aafe43221ea0d31395dee5386e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15f7a64633c9f07e2e70f08658436418
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 46, 82], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class PrimitiveOp_a0191fe6a1122ab6596c791553e81af5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 24, 92, 164], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a8b10b5916e58e41407558130a0a4041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0191fe6a1122ab6596c791553e81af5
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class PrimitiveOp_1abddc10690c7fb3fedd5221fef428ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 24, 184, 328], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_747075eea3a43a03c1808b6047f15c1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1abddc10690c7fb3fedd5221fef428ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class PrimitiveOp_7bf619cdf78e852feb559b5e72a5ba77(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 11, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8d1189e45391e41066af2b11ca34daf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bf619cdf78e852feb559b5e72a5ba77
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2548368275165558], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_9a2bb000481d11820bde8e63e87f442c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 17, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e14b942ace578ce54ccd5579942d90c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a2bb000481d11820bde8e63e87f442c
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_df9877dc51d8530c530af48a43e78dcc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 288, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 288, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d3b0aa731df951ab8966bd316a14afe9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df9877dc51d8530c530af48a43e78dcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57a22e9834dd91b2b43258cc81ba76ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06e3aeaa24fef692fbebc0f3db6bdcf8
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6d93c0a755c10350fd60d0534979829b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 88, 132], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09a0011df0e93ec687bc1207a9753ce5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d93c0a755c10350fd60d0534979829b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.042571865022182465], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_f5f2ac5a2cdb4bb1c7e477fd6827e878(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 20, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3c86651bb23b0319e53fb0cd3bc82482(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5f2ac5a2cdb4bb1c7e477fd6827e878
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1272b57e0e52d16007171347313276a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bccae2098479943af12544f16f40472c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1272b57e0e52d16007171347313276a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bccae2098479943af12544f16f40472c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1272b57e0e52d16007171347313276a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bccae2098479943af12544f16f40472c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1272b57e0e52d16007171347313276a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f1b226a372776cdfaffc0a165cc8ac25(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4185, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8fa184a81ce65969ff2caabf27ad82b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1b226a372776cdfaffc0a165cc8ac25
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fa184a81ce65969ff2caabf27ad82b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1b226a372776cdfaffc0a165cc8ac25
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fa184a81ce65969ff2caabf27ad82b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1b226a372776cdfaffc0a165cc8ac25
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fa184a81ce65969ff2caabf27ad82b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1b226a372776cdfaffc0a165cc8ac25
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fa184a81ce65969ff2caabf27ad82b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1b226a372776cdfaffc0a165cc8ac25
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5be8adaac2e5a603c7a01bbde3727b93(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4185, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[4185, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6243d6e3f406f7b0f8a8c9a4a0f71c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5be8adaac2e5a603c7a01bbde3727b93
    def get_inputs(self):
        return [
            paddle.uniform([4185, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6243d6e3f406f7b0f8a8c9a4a0f71c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5be8adaac2e5a603c7a01bbde3727b93
    def get_inputs(self):
        return [
            paddle.uniform([4185, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fa184a81ce65969ff2caabf27ad82b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1b226a372776cdfaffc0a165cc8ac25
    def get_inputs(self):
        return [
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4185, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_966012030699764a7cc0a9c8e8a699a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 19, 29], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_23401cd20a10336790ab0603090ffc0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_966012030699764a7cc0a9c8e8a699a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 19, 29], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.021267196163535118], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_583338ebcf3dbad8337070772726056a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 624, 20, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 624, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ddd6f702ec12fc9cdee5919c779dff17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_583338ebcf3dbad8337070772726056a
    def get_inputs(self):
        return [
            paddle.uniform([1, 624, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 624, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f957723d08a914a51f72ce7c3c47dc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8b788cb6625f4842bd8dcca7be8c085
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bccae2098479943af12544f16f40472c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1272b57e0e52d16007171347313276a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bccae2098479943af12544f16f40472c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1272b57e0e52d16007171347313276a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bccae2098479943af12544f16f40472c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1272b57e0e52d16007171347313276a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bccae2098479943af12544f16f40472c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1272b57e0e52d16007171347313276a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5fc13540db8c468ea0d8ee125772f421(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 256, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d450f708228b09ee75f03e05c13936c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fc13540db8c468ea0d8ee125772f421
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


class PrimitiveOp_90bdd5f80aeb40d33d559c31d86fc3f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7ded76f9bba2ca89fe9e75d4b79ca4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90bdd5f80aeb40d33d559c31d86fc3f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0d3890b2dd49a51fa89038161d787d38(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 68, 7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0823522fd303003086ba4026b5b35d91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3890b2dd49a51fa89038161d787d38
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15094605088233948], dtype='float32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()