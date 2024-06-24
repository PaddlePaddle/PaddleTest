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


class TestPrimitiveOp_9c86e2bdee257c9dba59081a8ebce952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092fc4de7b2e759352aca979a690cfef
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


class TestPrimitiveOp_882ac2b46a85ff341c5d6e272e11c64c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123c4c37f31fa937c21ca370a3424f4b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.34175336360931396], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_db0084c677d3e04909df20d4896c5ad2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_412d348f8b602df6f1332b51a43c0549
    def get_inputs(self):
        return [
            paddle.uniform([4, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.1975490152835846]], [[0.14006461203098297]], [[0.38203638792037964]], [[0.2991730868816376]]], dtype='float32').reshape([4, 1, 1]),
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


class TestPrimitiveOp_d841798d884035434e67c963d4c21887(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d467e8235e880f3c5a8f5baf9bacd88
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.25486618280410767], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_815214f5bd856c9a4bd6c920713360ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06afa0ac8681719902f3e86d0feb1b9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.9020241498947144]], [[0.909557044506073]], [[0.9281327724456787]], [[0.882320761680603]], [[0.957533061504364]], [[0.9401796460151672]], [[0.8982873558998108]], [[0.8895022869110107]], [[0.9026750326156616]], [[0.9215229749679565]], [[0.9120767712593079]], [[0.8973357081413269]], [[0.947052538394928]], [[0.9509029388427734]], [[0.9220102429389954]], [[0.8148493766784668]], [[0.9395149946212769]], [[0.9384528994560242]], [[0.90580815076828]], [[0.9223538637161255]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_8251dbd43c603b30990d514e20f6fc52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f226bf699b81a8340f1c4c0f15e3c537
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.2497674822807312]]], dtype='float32').reshape([1, 1, 1]),
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


class TestPrimitiveOp_2130e167d528c2ef6a495da19376d9b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f82bae9f01017bc84c73b0afb0ceb402
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22599859535694122], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_50cb05f78c615bc449db2a106ca03ed2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be9379a2ffd678c991404990a3f835ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 50, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3698897063732147], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_38e5e2fc464e9016c160b755585af711(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9afbcc6f57ba50a3da6628142dd38ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.07537981122732162], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_16b60cd92ba33a931be4cc5637ef923f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39670343df85f06e46d21b863131110d
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.756847620010376]], [[0.7256442308425903]], [[0.7928997874259949]], [[0.7176178693771362]], [[0.8461375832557678]], [[0.9047800898551941]], [[0.9056916236877441]], [[0.8323326110839844]], [[0.746349573135376]], [[0.844599723815918]], [[0.7069713473320007]], [[0.7190803289413452]], [[0.7653292417526245]], [[0.8206942677497864]], [[0.6785595417022705]], [[0.7115932703018188]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_85a61b432ac1425b28c108da436bd5ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21640bfeddd1cc679a8a6fae429cb547
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.06117178499698639, 0.1596771627664566]], [[0.4163835048675537, -0.3349427282810211]], [[-0.04870566725730896, -0.19619053602218628]], [[0.25524312257766724, 0.21017765998840332]], [[0.2934775948524475, 0.27044686675071716]], [[0.33601114153862, -0.16985705494880676]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_0c0b630f50e616652c474b671ee6910c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21640bfeddd1cc679a8a6fae429cb547
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[-0.023209571838378906, 0.41274580359458923]], [[0.4644279181957245, -0.3066585659980774]], [[-0.2569788694381714, -0.059861332178115845]], [[0.0648866519331932, 0.23688597977161407]], [[-0.12440935522317886, 0.12358363717794418]], [[0.053374744951725006, 0.11837655305862427]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class TestPrimitiveOp_d2f55b0f70a9e77d5f514cc13fe69065(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4992024e84675fd31e5997a5c142f74e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.06117178499698639, 0.1596771627664566]], [[0.4163835048675537, -0.3349427282810211]], [[-0.04870566725730896, -0.19619053602218628]], [[0.25524312257766724, 0.21017765998840332]], [[0.2934775948524475, 0.27044686675071716]], [[0.33601114153862, -0.16985705494880676]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.06117178499698639, 0.1596771627664566]], [[0.4163835048675537, -0.3349427282810211]], [[-0.04870566725730896, -0.19619053602218628]], [[0.25524312257766724, 0.21017765998840332]], [[0.2934775948524475, 0.27044686675071716]], [[0.33601114153862, -0.16985705494880676]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_d567a7abd02192143e0712874a0d357d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4992024e84675fd31e5997a5c142f74e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.023209571838378906, 0.41274580359458923]], [[0.4644279181957245, -0.3066585659980774]], [[-0.2569788694381714, -0.059861332178115845]], [[0.0648866519331932, 0.23688597977161407]], [[-0.12440935522317886, 0.12358363717794418]], [[0.053374744951725006, 0.11837655305862427]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[-0.023209571838378906, 0.41274580359458923]], [[0.4644279181957245, -0.3066585659980774]], [[-0.2569788694381714, -0.059861332178115845]], [[0.0648866519331932, 0.23688597977161407]], [[-0.12440935522317886, 0.12358363717794418]], [[0.053374744951725006, 0.11837655305862427]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class TestPrimitiveOp_7e853cb57600f2cfd36acf8dbc2c8e53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe310264e1b88b3efe1ed272b8463698
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.004999642726033926], [0.1525985151529312], [0.00826028361916542], [0.03614693880081177], [0.06356286257505417], [0.05337122082710266]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([[[0.1690986603498459], [0.3384711444377899], [0.019616106525063515], [0.07273853570222855], [0.11445915699005127], [0.18592406809329987]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_3b5becc5ae31eca47878ac10e5ff1b1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe310264e1b88b3efe1ed272b8463698
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.07064876705408096], [0.1723775416612625], [0.0183702539652586], [0.014816601760685444], [0.005392380058765411], [0.002189568942412734]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([[[0.1690986603498459], [0.3384711444377899], [0.019616106525063515], [0.07273853570222855], [0.11445915699005127], [0.18592406809329987]]], dtype='float32').reshape([1, 6, 1]),
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


class TestPrimitiveOp_8c5d131acc31d907b0326e621f55e116(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7852bc1b6dcc7869f2cc107389bc5f0b
    def get_inputs(self):
        return [
            paddle.to_tensor([2.213313579559326, 1.9694278240203857, 2.2759838104248047, 1.9948484897613525, 2.2433714866638184, 2.026952028274536, 2.2625842094421387, 2.3940014839172363, 2.1893672943115234, 1.9999949932098389, 2.039778232574463, 2.0384669303894043, 2.334479570388794, 1.9468075037002563, 1.8638880252838135, 2.090146541595459], dtype='float32').reshape([16]),
            paddle.to_tensor([0.5746708512306213, 0.5966348052024841, 0.7137987017631531, 0.5975824594497681, 0.994225800037384, 0.9249470829963684, 0.9756565690040588, 0.5262556076049805, 0.98907470703125, 0.8677127361297607, 0.5212010145187378, 0.7002993822097778, 0.556075930595398, 0.7877225875854492, 0.5620827674865723, 0.9226763844490051], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_ce5cb13583426ac43e0f2113827b87a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7852bc1b6dcc7869f2cc107389bc5f0b
    def get_inputs(self):
        return [
            paddle.to_tensor([1.9784406423568726, 1.894113302230835, 1.8887768983840942, 1.9572476148605347, 1.8944146633148193, 2.0069973468780518, 2.022692918777466, 2.200212001800537, 1.8431483507156372, 1.8886048793792725, 2.1080193519592285, 2.2595934867858887, 2.2477219104766846, 2.0040104389190674, 2.1124138832092285, 2.153306245803833], dtype='float32').reshape([16]),
            paddle.to_tensor([0.42532914876937866, 0.40336519479751587, 0.2862012982368469, 0.40241751074790955, 0.005774199031293392, 0.0750528946518898, 0.024343404918909073, 0.4737444221973419, 0.010925306007266045, 0.13228726387023926, 0.4787990152835846, 0.29970064759254456, 0.44392403960227966, 0.21227741241455078, 0.43791723251342773, 0.07732360810041428], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_e8fdfc58830e599c838d0b04ea045030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7852bc1b6dcc7869f2cc107389bc5f0b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5283538103103638, 0.48476213216781616, 0.5412911772727966, 0.49492931365966797, 0.560339093208313, 0.506363570690155, 0.5641860961914062, 0.5755487084388733, 0.5463961958885193, 0.4963148832321167, 0.5181130170822144, 0.5261846780776978, 0.5739914178848267, 0.4897376298904419, 0.49318045377731323, 0.5237575769424438], dtype='float32').reshape([16]),
            paddle.to_tensor([0.24020686745643616, 0.278904527425766, 0.024928631260991096, 0.16140449047088623, 0.4684695601463318, 0.10429177433252335, 0.2114018350839615, 0.011752157472074032, 0.300222784280777, 0.18888747692108154, 0.17155683040618896, 0.47473734617233276, 0.15425065159797668, 0.27774766087532043, 0.4100176692008972, 0.20129874348640442], dtype='float32').reshape([16]),
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


class TestPrimitiveOp_21bc4b793ce20f2219dad3fe493c354a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_210051582262298800fe8fc79767c3f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17315250635147095], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_d113a3e8e90edc429da76353065c60d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c63c960682dda3a4cada933be990e9d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3114614486694336], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_0dfdf2fab6d16ffbe600e85f3a711c34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88524ba379499f55c3ea410fbee25873
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.4053474962711334, 0.40568044781684875, 0.010601983405649662, 0.2857620418071747]]], dtype='float32').reshape([1, 1, 4]),
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


class TestPrimitiveOp_1ca927f6be4e73de4f73b22673d35586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce4ace322016d97bf695286595d16834
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3388667702674866], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_eab5e342ae5fe917aba40951192df91e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b0bf5dc6a91df258aef9a721530f24e
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.45048007369041443], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_9aeadc5f0a81e451b5eb0e843ccdffd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07c059a42edcf3de218e97da9fbac96f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.20757755637168884], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_6b29e8ec6f2baaec361ae126fde20ad9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c712740622f241d3603e5831e540ce30
    def get_inputs(self):
        return [
            paddle.uniform([3, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.4277102053165436]], [[0.4063163995742798]], [[0.1532394289970398]]], dtype='float32').reshape([3, 1, 1]),
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


class TestPrimitiveOp_8db403faddd3561773bfd3be9cc5d19e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a26dfc3e1ee7aecbc2a9fe0ad4767ca7
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 100, 152], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.23824641108512878], dtype='float32').reshape([1]),
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


class PrimitiveOp_f4b3ec1b31e8eea84ffc026270b6c063(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1758, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1758, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bbc6c3722ce332ba822c90c8eb4a02b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4b3ec1b31e8eea84ffc026270b6c063
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbc6c3722ce332ba822c90c8eb4a02b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4b3ec1b31e8eea84ffc026270b6c063
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbc6c3722ce332ba822c90c8eb4a02b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4b3ec1b31e8eea84ffc026270b6c063
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbc6c3722ce332ba822c90c8eb4a02b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4b3ec1b31e8eea84ffc026270b6c063
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbc6c3722ce332ba822c90c8eb4a02b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4b3ec1b31e8eea84ffc026270b6c063
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_303dc317ed44a5ae54c57853c2ab201d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1758, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1758, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_76440f7dcbb48ff99ea80510bc390a53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_303dc317ed44a5ae54c57853c2ab201d
    def get_inputs(self):
        return [
            paddle.uniform([1758, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76440f7dcbb48ff99ea80510bc390a53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_303dc317ed44a5ae54c57853c2ab201d
    def get_inputs(self):
        return [
            paddle.uniform([1758, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbc6c3722ce332ba822c90c8eb4a02b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4b3ec1b31e8eea84ffc026270b6c063
    def get_inputs(self):
        return [
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1758, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f6ae2d5534afea8d3c4cbdd26a128e0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fb038135fe2fa8372d9386ebb79df7d
    def get_inputs(self):
        return [
            paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


class TestPrimitiveOp_521644c39928f0285e40219f10415e31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c12ca22a303ef482473d6959339461cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.13108214735984802], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_79c641c2ca824203d00b8797dd5c89a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53d856c85717ca4bec7292dee1ab55f9
    def get_inputs(self):
        return [
            paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


class TestPrimitiveOp_a659cf78d38fcc16f5d1a77c853c1b89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f4316e09ac823e38388127c20d36206
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.24968935549259186], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_1ac91f66ce82e724c7fb81596f0a46b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61e2a73b85b68cf70c72f6ba6e376320
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.049740761518478394], [0.0], [0.0], [0.0], [0.0], [0.0], [0.1452496200799942]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_d80d1200d5057ff40440ba8b2af1efe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61e2a73b85b68cf70c72f6ba6e376320
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.32681986689567566], [0.18811313807964325], [-0.1081421971321106], [0.04511937499046326], [-0.06539356708526611], [-0.34937381744384766], [-0.08356040716171265], [0.2154596745967865], [-0.015725374221801758]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.2049010694026947], [-0.23399370908737183], [0.07450598478317261], [-0.06351912766695023], [-0.07621455192565918], [-0.27520930767059326], [-0.049346789717674255], [-0.3282959461212158], [0.2913358211517334]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_bed3d72fef22c839efceca3cc2bfbf34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61e2a73b85b68cf70c72f6ba6e376320
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.18039487302303314], [-0.1867675483226776], [0.4438837766647339], [-0.26366037130355835], [0.219099760055542], [0.2465997189283371], [0.16789889335632324], [-0.1690690964460373], [-0.2615792155265808]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.2627970278263092], [0.09249058365821838], [0.20684346556663513], [-0.30344492197036743], [-0.08254097402095795], [-0.4421014189720154], [0.07670104503631592], [0.0029647648334503174], [0.269756555557251]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_000f69e2a847f65c43e247b484cfe327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61e2a73b85b68cf70c72f6ba6e376320
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.18039487302303314], [0.18811313807964325], [0.4438837766647339], [0.04511937499046326], [0.24010060727596283], [0.2465997189283371], [0.18878795206546783], [0.2154596745967865], [-0.0022359490394592285]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.09970888495445251], [0.10120543837547302], [0.23160868883132935], [-0.005910724401473999], [0.1252661943435669], [-0.2700882852077484], [0.07670104503631592], [0.0029647648334503174], [0.41584277153015137]], dtype='float32').reshape([9, 1]),
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


class TestPrimitiveOp_bd6bf28abb952a42aba6b7d0ce12f930(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32089dd975aa5ae09d0e77a4bb9afc16
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1917666643857956], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bd6bf28abb952a42aba6b7d0ce12f930(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32089dd975aa5ae09d0e77a4bb9afc16
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1917666643857956], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_b38d89208f708e89ccf44732cca68d86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96bdec893315e84077b46ee1d2a7eea7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4411681294441223], dtype='float32').reshape([1]),
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


class PrimitiveOp_04a101728cd408847ff64d9655a99ec1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5593, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[5593, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_130d85f4776a6ed3e938ce8d95270d9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04a101728cd408847ff64d9655a99ec1
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_130d85f4776a6ed3e938ce8d95270d9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04a101728cd408847ff64d9655a99ec1
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_130d85f4776a6ed3e938ce8d95270d9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04a101728cd408847ff64d9655a99ec1
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_130d85f4776a6ed3e938ce8d95270d9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04a101728cd408847ff64d9655a99ec1
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_130d85f4776a6ed3e938ce8d95270d9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04a101728cd408847ff64d9655a99ec1
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_800c0f3e12d3bbdf87d34c0415d1ed6d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5593, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[5593, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_32501893831b2c63ac85968490ccbedf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_800c0f3e12d3bbdf87d34c0415d1ed6d
    def get_inputs(self):
        return [
            paddle.uniform([5593, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32501893831b2c63ac85968490ccbedf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_800c0f3e12d3bbdf87d34c0415d1ed6d
    def get_inputs(self):
        return [
            paddle.uniform([5593, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_130d85f4776a6ed3e938ce8d95270d9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04a101728cd408847ff64d9655a99ec1
    def get_inputs(self):
        return [
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5593, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_87a4e820ff4bb0b38226730e2d540772(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77abb17d1cf32ac968d3cf6df91220de
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.24615374207496643], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_796f9d72579ce3ecb7c48a79b6e802c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a44734e17f450d9c0dbb1333fb739ac0
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.24486394226551056], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_ad466da634964b2b8b38060079fa2104(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8425318598747253], dtype='float32').reshape([1]),
            paddle.to_tensor([0.19734811782836914], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8eb0e27273c18d3a6ff565c85668ba60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7325313687324524], dtype='float32').reshape([1]),
            paddle.to_tensor([0.3503933548927307], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_895c92d7f0e145cc470dcb0223bfedbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22aec529d4321b5699d224d6ffb96e74
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4716094434261322], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_569524f0ac6322708e49dda3f1231ebe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743da1a40e2d519d599bc44cb5426535
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4868690073490143], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_424e4b54d7c730ad2a43a2d257ecb90f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.10491561889648438, -0.18058955669403076, -0.07093271613121033, -0.11325669288635254, -0.44685986638069153, -0.20697996020317078], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.25567948818206787, 0.06439828872680664, -0.23553410172462463, -0.4003322422504425, -0.1100035011768341, -0.04974937438964844], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_540b4f50b23b1c51c123d006e6b90f68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.026824772357940674, -0.011629658751189709, 0.016707073897123337, 0.04534030705690384, 0.04915614798665047, 0.010297123342752457], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_591eaa47fddb404d437b7854bfdc89f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, -0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_aa695c8a66267032bd173b10115db1a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.15271055698394775, 0.0, 0.09368011355400085, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, 0.06439828872680664, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_0de89ceb808dc959680b7993de02ee3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.09959647059440613, 0.05746108293533325, 0.024996191263198853, -0.01376459002494812, -0.2947706878185272, -0.12244720757007599], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.25567948818206787, 0.32966962456703186, -0.16224297881126404, -0.3088887929916382, 0.27184319496154785, 0.27993857860565186], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_4134d1da74091721ea1bf01bfc24dbac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1314726620912552, 0.007713735103607178, -0.13027089834213257, -0.16256336867809296, 0.1488959640264511, -0.18264752626419067], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1314726620912552, 0.007713735103607178, -0.13027089834213257, -0.16256336867809296, 0.1488959640264511, -0.18264752626419067], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_f96081f008d201773727079abcfbf8b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06926202774047852, 0.0405440479516983, 0.0031562447547912598, -0.2646448612213135, 0.15198823809623718, 0.09625422954559326], dtype='float32').reshape([6]),
            paddle.to_tensor([0.06926202774047852, 0.0405440479516983, 0.0031562447547912598, -0.2646448612213135, 0.15198823809623718, 0.09625422954559326], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_4e359b62640c92fa952b17d156ee06d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.158029705286026, 0.238050639629364, 0.18960902094841003, 0.09949210286140442, 0.1520891785621643, 0.08453275263309479], dtype='float32').reshape([6]),
            paddle.to_tensor([0.158029705286026, 0.238050639629364, 0.18960902094841003, 0.09949210286140442, 0.1520891785621643, 0.08453275263309479], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_47f87ec16c7420db32b1b67736f0050f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 0.32966962456703186, 0.0732911229133606, 0.09144344925880432, 0.38184672594070435, 0.3296879529953003], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, 0.32966962456703186, 0.0732911229133606, 0.09144344925880432, 0.38184672594070435, 0.3296879529953003], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_b792b16dc5bc475b1a51d19b8f2c40c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.6501924991607666, 0.6423712372779846, 0.05975647643208504, -0.48799392580986023, -0.7452673316001892, -0.3194734752178192], dtype='float32').reshape([6]),
            paddle.to_tensor([1.604284644126892, 1.5849864482879639, 0.14744310081005096, -1.2040759325027466, -1.8388721942901611, -0.7882686853408813], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_9f1688629ae7437eb5459d647bd4292f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5105462670326233, 0.5044965744018555, 0.008733730763196945, 0.3701111674308777, 0.5781393647193909, 0.20117008686065674], dtype='float32').reshape([6]),
            paddle.to_tensor([1.0430938005447388, 1.018149733543396, 0.008810680359601974, 0.5875817537307739, 1.3704513311386108, 0.25183093547821045], dtype='float32').reshape([6]),
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


class PrimitiveOp_419838f220d59f4e663ced44e3d03c52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1763, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1763, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7515cb80036d231c5fdcf6455c13fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_419838f220d59f4e663ced44e3d03c52
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7515cb80036d231c5fdcf6455c13fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_419838f220d59f4e663ced44e3d03c52
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7515cb80036d231c5fdcf6455c13fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_419838f220d59f4e663ced44e3d03c52
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7515cb80036d231c5fdcf6455c13fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_419838f220d59f4e663ced44e3d03c52
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7515cb80036d231c5fdcf6455c13fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_419838f220d59f4e663ced44e3d03c52
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_089edf0d62cea4b9a026501f056dc950(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1763, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1763, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ba6b107c93c835f171f313b1cde9bc3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_089edf0d62cea4b9a026501f056dc950
    def get_inputs(self):
        return [
            paddle.uniform([1763, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba6b107c93c835f171f313b1cde9bc3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_089edf0d62cea4b9a026501f056dc950
    def get_inputs(self):
        return [
            paddle.uniform([1763, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7515cb80036d231c5fdcf6455c13fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_419838f220d59f4e663ced44e3d03c52
    def get_inputs(self):
        return [
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1763, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_9c56190d44702017ecbc047daa0bfdda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f74d0ba23ea63a3e855d3b191829cd80
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.46193239092826843], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_529983c0ed5129521ff59f136b99e229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a925e822f3f30e75c3ce588dc94246f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.28901633620262146], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_dace57517f170b2eabd859805e25bf28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49df513be1b8d3a4da5292a4568c65d1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.2834267616271973, 1.9475374221801758, 1.9306038618087769, 2.1778082847595215, 2.1797707080841064, 2.006352424621582, 2.2974331378936768, 2.184471607208252, 2.006229877471924, 1.9248852729797363, 2.1701712608337402, 2.1094906330108643, 2.056558609008789, 2.187903881072998, 2.147904396057129, 1.9835880994796753, 2.387515068054199, 1.9674217700958252, 1.884556531906128, 2.200183391571045, 1.9284087419509888, 2.000828266143799, 2.1119980812072754, 2.121765375137329], dtype='float32').reshape([24]),
            paddle.to_tensor([0.9669412970542908, 0.8978354930877686, 0.8152099251747131, 0.5462236404418945, 0.8771392703056335, 0.679118275642395, 0.7972595691680908, 0.6741753816604614, 0.7036166191101074, 0.6871111392974854, 0.7405611872673035, 0.803400993347168, 0.9205933213233948, 0.6569225788116455, 0.5710830092430115, 0.5544105172157288, 0.7950307130813599, 0.8525682687759399, 0.8444293737411499, 0.593220591545105, 0.6057742834091187, 0.9977433681488037, 0.7086058259010315, 0.6081423759460449], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_a2fa582866f7ad50c0a061b7950d450e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49df513be1b8d3a4da5292a4568c65d1
    def get_inputs(self):
        return [
            paddle.to_tensor([1.9443585872650146, 1.8843632936477661, 2.3338170051574707, 2.1591877937316895, 2.19769024848938, 2.2046902179718018, 1.9113688468933105, 2.0742058753967285, 2.0201146602630615, 1.9687464237213135, 2.163201332092285, 2.0628907680511475, 2.059060573577881, 2.096897602081299, 2.2119154930114746, 2.073439121246338, 2.087416648864746, 2.0209898948669434, 2.013093948364258, 2.137061357498169, 2.1214048862457275, 2.169085741043091, 1.9357800483703613, 2.0269126892089844], dtype='float32').reshape([24]),
            paddle.to_tensor([0.03305868059396744, 0.10216449201107025, 0.18479007482528687, 0.45377635955810547, 0.12286072969436646, 0.320881724357605, 0.20274046063423157, 0.3258246183395386, 0.29638341069221497, 0.31288883090019226, 0.25943881273269653, 0.19659900665283203, 0.07940669357776642, 0.3430773913860321, 0.4289169907569885, 0.44558948278427124, 0.20496925711631775, 0.14743173122406006, 0.1555706411600113, 0.406779408454895, 0.39422574639320374, 0.0022566039115190506, 0.2913941740989685, 0.3918576240539551], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_0815e1eb24a89b3e224d05b3887540c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49df513be1b8d3a4da5292a4568c65d1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5680543780326843, 0.48527079820632935, 0.5012784004211426, 0.5423396825790405, 0.5454930663108826, 0.5174988508224487, 0.5547905564308167, 0.5371360778808594, 0.502586305141449, 0.48465222120285034, 0.5420907735824585, 0.5250822901725769, 0.5141893625259399, 0.5391703844070435, 0.5438399910926819, 0.5059062242507935, 0.5815010070800781, 0.4938298463821411, 0.47613829374313354, 0.5436266660690308, 0.5011231899261475, 0.5003019571304321, 0.5151622891426086, 0.5211491584777832], dtype='float32').reshape([24]),
            paddle.to_tensor([0.11140142381191254, 0.41462281346321106, 0.1958979219198227, 0.36391329765319824, 0.19545233249664307, 0.30471816658973694, 0.33503809571266174, 0.0615343414247036, 0.18028651177883148, 0.43628066778182983, 0.009729217737913132, 0.01876985840499401, 0.36866724491119385, 0.31910648941993713, 0.19152416288852692, 0.3655022382736206, 0.4165481626987457, 0.11802870780229568, 0.13556380569934845, 0.0403435043990612, 0.11016726493835449, 0.3124121129512787, 0.268606573343277, 0.19298312067985535], dtype='float32').reshape([24]),
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


class TestPrimitiveOp_31da12cd1521b73ac816bdc2205a3512(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_550fd531e9b0787d351dc130b2e59d13
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24528706073760986], [0.2425270825624466]]], dtype='float32').reshape([1, 2, 1]),
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


class TestPrimitiveOp_3b2f22f93d96944575015e56400002c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd27cb57db362f26d74af4bb48557b9b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1132647916674614], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_b7bf0b4162a8d0904e217e9dee0e362c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90997f531f3da2136342d353b81fee34
    def get_inputs(self):
        return [
            paddle.to_tensor([2.1882402896881104, 2.052551746368408, 1.9094576835632324, 1.9341490268707275], dtype='float32').reshape([4]),
            paddle.to_tensor([0.8805320858955383, 0.5868434906005859, 0.709433913230896, 0.7869852781295776], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_ebc3bd831678dae153e41f6d4d6bc52b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90997f531f3da2136342d353b81fee34
    def get_inputs(self):
        return [
            paddle.to_tensor([2.070204973220825, 2.035569667816162, 2.0479705333709717, 2.227886438369751], dtype='float32').reshape([4]),
            paddle.to_tensor([0.11946789920330048, 0.4131564795970917, 0.2905661165714264, 0.21301473677158356], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_7d16549b4dfe8bf83eec018e5974f05c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90997f531f3da2136342d353b81fee34
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5435346961021423, 0.5113838911056519, 0.487426221370697, 0.499179869890213], dtype='float32').reshape([4]),
            paddle.to_tensor([0.18324348330497742, 0.36732539534568787, 0.4216848909854889, 0.3857714533805847], dtype='float32').reshape([4]),
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


class TestPrimitiveOp_3f2e41dfae7aa2b4cabea459d4e43452(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0399b6b5890bd2e28b6c4105967fdbad
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4902363419532776], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_63ed0188ec89a1c5e0b23f01bad3f6a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0b9f03b1162d46c3768e32b17f3c7dd
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.12959706783294678]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.03925694525241852]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_9c3d0b8b56e55e64321377179aefa77a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0b9f03b1162d46c3768e32b17f3c7dd
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.10576975345611572]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.016650795936584473]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_fef01582f459817c4b8e75f6f2cf6f77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0b9f03b1162d46c3768e32b17f3c7dd
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.07316523790359497]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.03925694525241852]], dtype='float32').reshape([1, 1]),
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


class TestPrimitiveOp_f54facc200739daa946749e6b430f81f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b2780f69624fff122ed8c64c70bbb6d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_54a76c2b87b26958cbbbaaa672ad0a58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b2780f69624fff122ed8c64c70bbb6d
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.08144965767860413], [0.1037202924489975], [-0.053121715784072876], [-0.03630611300468445], [0.0006935745477676392], [-0.11986371874809265]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.23791563510894775], [-0.2085690200328827], [-0.1481981873512268], [-0.005472943186759949], [-0.32715344429016113], [-0.11203506588935852]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_8f938bab08846d4cca5d271ea2b7030b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b2780f69624fff122ed8c64c70bbb6d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2684990167617798], [-0.1810915172100067], [0.06229901313781738], [0.15417543053627014], [0.13329753279685974], [0.239049032330513]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.10433448851108551], [0.09243687987327576], [0.32840049266815186], [0.314190536737442], [-0.3308977782726288], [0.26026058197021484]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_70a9deb37dffcff9a6f571c66962fa0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b2780f69624fff122ed8c64c70bbb6d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3693501353263855], [0.2343365103006363], [0.08228719234466553], [0.2161482721567154], [0.3955715298652649], [0.3294372260570526]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.10433448851108551], [0.09243687987327576], [0.32840049266815186], [0.314190536737442], [-0.19108976423740387], [0.26026058197021484]], dtype='float32').reshape([6, 1]),
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


class TestPrimitiveOp_a9c2d7235957d7db49d22a5399a24985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a456f0aa8c14d027aa36c5d72198f1a
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


class TestPrimitiveOp_9da774ae239aa10620c05ca8090c946e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b79dd7060ce3ed8be9a67757b148a4d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.7679421305656433]], [[0.7329837679862976]], [[0.8145518898963928]], [[0.7124038934707642]], [[0.7622596025466919]], [[0.7966616749763489]], [[0.674647867679596]], [[0.8319457769393921]], [[0.9780449271202087]], [[0.7808970212936401]], [[0.8721251487731934]], [[0.8689228892326355]], [[0.7657139897346497]], [[0.8143513202667236]], [[0.8331177234649658]], [[0.8438591957092285]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


class TestPrimitiveOp_d43f465e16372112b52c131d0c139082(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd3206ed2fe7ac43381c2bbb0e5d2f9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 10, 15], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.29324060678482056], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_4aff7e8430e44e9cd256e39acef2864f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5ea2b5f2f1308e8175fb854b8b66069
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24650290608406067]]], dtype='float32').reshape([1, 1, 1]),
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


class TestPrimitiveOp_76c87e7b16f3b9ed0e22a044e716da66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d835d8df3c2e970a3a833e74afa70962
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.12372072786092758], dtype='float32').reshape([1]),
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


class PrimitiveOp_638ff3a72c2caea01b6cb40bfc0ae30f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2076, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2076, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df06b55fdd2bd85ebfb169fdd3daaa0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_638ff3a72c2caea01b6cb40bfc0ae30f
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df06b55fdd2bd85ebfb169fdd3daaa0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_638ff3a72c2caea01b6cb40bfc0ae30f
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df06b55fdd2bd85ebfb169fdd3daaa0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_638ff3a72c2caea01b6cb40bfc0ae30f
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df06b55fdd2bd85ebfb169fdd3daaa0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_638ff3a72c2caea01b6cb40bfc0ae30f
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df06b55fdd2bd85ebfb169fdd3daaa0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_638ff3a72c2caea01b6cb40bfc0ae30f
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f235dbef43b063115920abae94c1aea2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2076, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2076, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d113cdfa9b8ec930c0f57cf1a1578ae2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f235dbef43b063115920abae94c1aea2
    def get_inputs(self):
        return [
            paddle.uniform([2076, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d113cdfa9b8ec930c0f57cf1a1578ae2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f235dbef43b063115920abae94c1aea2
    def get_inputs(self):
        return [
            paddle.uniform([2076, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df06b55fdd2bd85ebfb169fdd3daaa0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_638ff3a72c2caea01b6cb40bfc0ae30f
    def get_inputs(self):
        return [
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2076, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_4ed286d5a25ef6d7ff64323fc1b9a663(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.720988929271698], dtype='float32').reshape([1]),
            paddle.to_tensor([0.08674124628305435], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b56cea2d3596b1d457d6a2f8f80e5ae0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8319710493087769], dtype='float32').reshape([1]),
            paddle.to_tensor([0.4475049376487732], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9d48f5a25f15bb479d55ae77d5f78b22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.883206844329834], dtype='float32').reshape([1]),
            paddle.to_tensor([0.00015123066259548068], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7d0ee6d5b4e7bffaed6d959b0205c92b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.6733518838882446], dtype='float32').reshape([1]),
            paddle.to_tensor([0.3665471374988556], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_70f6b622be7caed1e49ffcec1868e0ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7482179403305054], dtype='float32').reshape([1]),
            paddle.to_tensor([0.44122251868247986], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_958b2c127edeee0dc6201f2a13cb46c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.759205162525177], dtype='float32').reshape([1]),
            paddle.to_tensor([0.3463875651359558], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5cbd3d661259983913f34364ac37d628(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7613592147827148], dtype='float32').reshape([1]),
            paddle.to_tensor([0.4251581132411957], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b272b7355ec7817b6fc872433a2aa4b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.6719062328338623], dtype='float32').reshape([1]),
            paddle.to_tensor([0.3425384759902954], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_edbe694f3b43ab5b7d08af5bff5de0da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7999907732009888], dtype='float32').reshape([1]),
            paddle.to_tensor([0.0037448795046657324], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_5e9ceb6211ee06ecb487e55e6b8063f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90412ca909e2194393aa35bb44fdba24
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.022642318159341812], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5e9ceb6211ee06ecb487e55e6b8063f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90412ca909e2194393aa35bb44fdba24
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.022642318159341812], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_523d868555f3b704ce0ec905b4fad36e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d7f35f039e7ad2049eb45bad5e4b525
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.19212619960308075], dtype='float32').reshape([1]),
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


class PrimitiveOp_c91e6c1348d437bce7e0132cb54ed5c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4642, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4642, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_21a69db89907c42ed793025b3821521a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c91e6c1348d437bce7e0132cb54ed5c5
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21a69db89907c42ed793025b3821521a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c91e6c1348d437bce7e0132cb54ed5c5
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21a69db89907c42ed793025b3821521a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c91e6c1348d437bce7e0132cb54ed5c5
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21a69db89907c42ed793025b3821521a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c91e6c1348d437bce7e0132cb54ed5c5
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21a69db89907c42ed793025b3821521a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c91e6c1348d437bce7e0132cb54ed5c5
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b04a799b808ba231f8e21fc53b922547(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4642, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[4642, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1388193658e30f01477af637a012461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b04a799b808ba231f8e21fc53b922547
    def get_inputs(self):
        return [
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d1388193658e30f01477af637a012461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b04a799b808ba231f8e21fc53b922547
    def get_inputs(self):
        return [
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21a69db89907c42ed793025b3821521a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c91e6c1348d437bce7e0132cb54ed5c5
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_144d80a39fdd41b22d6db9a8fc956fa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84ffd7f472527bec19f44199ce5cac64
    def get_inputs(self):
        return [
            paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.14978468418121338]], [[0.13296274840831757]], [[0.039421454071998596]], [[0.1589626669883728]], [[0.23054076731204987]], [[0.48784351348876953]]], dtype='float32').reshape([6, 1, 1]),
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


class PrimitiveOp_9360436d98659b83e67b72d6a2189e71(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1047, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1047, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ea6f4198323304679b24447930e451c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9360436d98659b83e67b72d6a2189e71
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ea6f4198323304679b24447930e451c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9360436d98659b83e67b72d6a2189e71
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ea6f4198323304679b24447930e451c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9360436d98659b83e67b72d6a2189e71
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ea6f4198323304679b24447930e451c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9360436d98659b83e67b72d6a2189e71
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ea6f4198323304679b24447930e451c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9360436d98659b83e67b72d6a2189e71
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d6f73ed7b179d8e81564e77aa74e6d99(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1047, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1047, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_469c3905788816677b116c6e3ae60247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6f73ed7b179d8e81564e77aa74e6d99
    def get_inputs(self):
        return [
            paddle.uniform([1047, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_469c3905788816677b116c6e3ae60247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6f73ed7b179d8e81564e77aa74e6d99
    def get_inputs(self):
        return [
            paddle.uniform([1047, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ea6f4198323304679b24447930e451c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9360436d98659b83e67b72d6a2189e71
    def get_inputs(self):
        return [
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1047, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_6a9e630cdfbc13f30e9ee997ebfb3156(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d431583dbce2806d5aea6fac77553e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.006989086978137493], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_4654c75ffa4c194cb6b928c1ed249fd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71914385028c60457327f20b50af0ff7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.08329835534095764]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.030602753162384033], [0.0], [0.0], [0.0], [0.023834317922592163]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_2b220cd0de2617fb1b9b725cd6d81035(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71914385028c60457327f20b50af0ff7
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.027636408805847168], [-0.05974382162094116], [-0.09686454385519028], [-0.14594735205173492], [0.08329835534095764]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.383173406124115], [-0.1104418933391571], [-0.4333980679512024], [0.044965118169784546], [0.3328261375427246]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_4e1b162d95f3b5f23dd4b4bac8441fbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71914385028c60457327f20b50af0ff7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11729046702384949], [-0.053922802209854126], [0.010135218501091003], [-0.04130584001541138], [0.40391606092453003]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.06556153297424316], [0.30815237760543823], [-0.028024911880493164], [-0.2581019997596741], [0.023834317922592163]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_5dddcc44a20c4c4d9334ce14161fab3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71914385028c60457327f20b50af0ff7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.24695812165737152], [0.06564271450042725], [0.040128856897354126], [0.042540401220321655], [0.40391606092453003]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.4181321859359741], [0.30815237760543823], [-0.028024911880493164], [0.044965118169784546], [0.3328261375427246]], dtype='float32').reshape([5, 1]),
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


class TestPrimitiveOp_bdd6e2d98739fddc18309c279f076e37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdbb53a9fcb2ba0cbe12e542d72858bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 76, 116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1708012968301773], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_df08aba6a3d361131f543e73af2581ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08cb7f26b76ddd9a4c61c34b99be7b82
    def get_inputs(self):
        return [
            paddle.to_tensor([0.01822187751531601], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_2f3ad1a3c018a397ecc6a168f5173f4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fcf03a6086130ae9486abb2f881899c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.33843323588371277], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2f3ad1a3c018a397ecc6a168f5173f4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fcf03a6086130ae9486abb2f881899c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.33843323588371277], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_190909bab94391d691860d266e40c290(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e955f135d79f9cab1a2d7bece733a436
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3248838484287262], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_b780c15895632865073c24c237ac208c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7138806c53fe65047584c00ade942317
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.38998275995254517], dtype='float32').reshape([1]),
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


class PrimitiveOp_4e1962348bd8fc6c05b13e95e293f7cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2359, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2359, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5cacd454ebb311dae7d4eee64a159821(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e1962348bd8fc6c05b13e95e293f7cd
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5cacd454ebb311dae7d4eee64a159821(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e1962348bd8fc6c05b13e95e293f7cd
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5cacd454ebb311dae7d4eee64a159821(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e1962348bd8fc6c05b13e95e293f7cd
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5cacd454ebb311dae7d4eee64a159821(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e1962348bd8fc6c05b13e95e293f7cd
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5cacd454ebb311dae7d4eee64a159821(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e1962348bd8fc6c05b13e95e293f7cd
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e16eef2819dae0770cf2368d92307798(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2359, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2359, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_284855e191458ac27da73ace0bae1a34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e16eef2819dae0770cf2368d92307798
    def get_inputs(self):
        return [
            paddle.uniform([2359, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_284855e191458ac27da73ace0bae1a34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e16eef2819dae0770cf2368d92307798
    def get_inputs(self):
        return [
            paddle.uniform([2359, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5cacd454ebb311dae7d4eee64a159821(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e1962348bd8fc6c05b13e95e293f7cd
    def get_inputs(self):
        return [
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2359, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_efec61f42d5f8d2b8ce2922af31d8d78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3049, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3049, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2dba4c82838678f84407ae44b9ad6f6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efec61f42d5f8d2b8ce2922af31d8d78
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2dba4c82838678f84407ae44b9ad6f6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efec61f42d5f8d2b8ce2922af31d8d78
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2dba4c82838678f84407ae44b9ad6f6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efec61f42d5f8d2b8ce2922af31d8d78
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2dba4c82838678f84407ae44b9ad6f6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efec61f42d5f8d2b8ce2922af31d8d78
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2dba4c82838678f84407ae44b9ad6f6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efec61f42d5f8d2b8ce2922af31d8d78
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2ff240ace9ca861cd7ed142569a32803(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3049, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[3049, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d5345913e5e663a07f2279ee2091f217(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ff240ace9ca861cd7ed142569a32803
    def get_inputs(self):
        return [
            paddle.uniform([3049, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d5345913e5e663a07f2279ee2091f217(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ff240ace9ca861cd7ed142569a32803
    def get_inputs(self):
        return [
            paddle.uniform([3049, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2dba4c82838678f84407ae44b9ad6f6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efec61f42d5f8d2b8ce2922af31d8d78
    def get_inputs(self):
        return [
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3049, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_abc924adf297024e0d6c61fe195f5179(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3806, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[3806, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_701aba56769825f7bf6cb7ac9e49be07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abc924adf297024e0d6c61fe195f5179
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_701aba56769825f7bf6cb7ac9e49be07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abc924adf297024e0d6c61fe195f5179
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_701aba56769825f7bf6cb7ac9e49be07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abc924adf297024e0d6c61fe195f5179
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_701aba56769825f7bf6cb7ac9e49be07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abc924adf297024e0d6c61fe195f5179
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_701aba56769825f7bf6cb7ac9e49be07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abc924adf297024e0d6c61fe195f5179
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cd0f01df8bc7cccc544223faa1be26cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3806, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[3806, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3283930579d757123d538653b8b5bb3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd0f01df8bc7cccc544223faa1be26cb
    def get_inputs(self):
        return [
            paddle.uniform([3806, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3283930579d757123d538653b8b5bb3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd0f01df8bc7cccc544223faa1be26cb
    def get_inputs(self):
        return [
            paddle.uniform([3806, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_701aba56769825f7bf6cb7ac9e49be07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abc924adf297024e0d6c61fe195f5179
    def get_inputs(self):
        return [
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3806, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_91743f140acd0ee315b9cbd97942d531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f904b4249c94af8f5bd043b44f301a7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.026380253955721855], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_91743f140acd0ee315b9cbd97942d531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f904b4249c94af8f5bd043b44f301a7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.026380253955721855], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_dbe5b5e83f5cbe3ff91fa08aeb5f5c40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a20ef40aebeb098fea9c3756d8eee753
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.49490052461624146], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_ec2cb3082baca95398883482a1b9bce9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06afa0ac8681719902f3e86d0feb1b9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.9119775295257568]], [[0.8058251142501831]], [[0.8422892689704895]], [[0.6895846128463745]], [[0.8328993320465088]], [[0.8175663948059082]], [[0.7157836556434631]], [[0.855390727519989]], [[0.8453803658485413]], [[0.8411725163459778]], [[0.8410471081733704]], [[0.917742133140564]], [[0.798347532749176]], [[0.9312773942947388]], [[0.8454999327659607]], [[0.856065034866333]], [[0.6812049746513367]], [[0.9054616689682007]], [[0.86566162109375]], [[0.8944621682167053]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_1914d8379d5bc39653dffabc09bd3918(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1e5c99c02e707f89895497c2deb623a
    def get_inputs(self):
        return [
            paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


class TestPrimitiveOp_5f4162fb1095f28874365c8bdc52b0d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08cb7f26b76ddd9a4c61c34b99be7b82
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3429355323314667], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_e102a18b95c3b61c1912953748c6ab4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf95189c0e8c270fd0d242cc8a5f9adc
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.322330504655838], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_28e47a573da66942d39ce4529198caa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5edaa4c79b377a22877dbe9049949df9
    def get_inputs(self):
        return [
            paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.342254102230072]], [[0.04157651588320732]]], dtype='float32').reshape([2, 1, 1]),
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


class TestPrimitiveOp_2e271df3239e171c40f36e6087e6bb33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b860cca7d74e41392aecd7a90ee59238
    def get_inputs(self):
        return [
            paddle.to_tensor([1.9969403743743896, 2.232224225997925, 2.1616365909576416, 2.382333755493164, 1.9233232736587524, 2.1826717853546143, 2.188194751739502, 2.2212114334106445, 1.9976115226745605, 1.9273639917373657, 2.208949089050293, 1.9189504384994507, 2.073084831237793, 1.8286184072494507, 2.136892080307007, 1.8656898736953735, 1.8957819938659668, 1.969542384147644, 2.1494929790496826, 2.1369214057922363], dtype='float32').reshape([20]),
            paddle.to_tensor([0.6676052808761597, 0.9447514414787292, 0.56711745262146, 0.8670538663864136, 0.7906093597412109, 0.9332652688026428, 0.7129138708114624, 0.9641093015670776, 0.8437008261680603, 0.5791691541671753, 0.7914769649505615, 0.539609432220459, 0.5366583466529846, 0.7358188629150391, 0.6439612507820129, 0.5326361656188965, 0.6670117378234863, 0.5232236981391907, 0.7371448278427124, 0.7546494603157043], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_c782f171c855645eed70898b3ee523f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b860cca7d74e41392aecd7a90ee59238
    def get_inputs(self):
        return [
            paddle.to_tensor([2.3048977851867676, 2.116800308227539, 2.23038649559021, 2.011624813079834, 2.2954282760620117, 1.9638127088546753, 2.1289992332458496, 2.2716681957244873, 1.912353754043579, 2.214439868927002, 1.949325680732727, 2.3215630054473877, 2.042908191680908, 2.256613254547119, 2.226163625717163, 2.172593593597412, 2.1633942127227783, 2.1945648193359375, 2.174997329711914, 2.17669939994812], dtype='float32').reshape([20]),
            paddle.to_tensor([0.3323947489261627, 0.055248551070690155, 0.43288254737854004, 0.13294614851474762, 0.20939065515995026, 0.06673471629619598, 0.2870861291885376, 0.03589070588350296, 0.1562991589307785, 0.4208308756351471, 0.20852304995059967, 0.460390567779541, 0.4633416533470154, 0.2641811668872833, 0.35603874921798706, 0.46736380457878113, 0.3329882323741913, 0.4767763018608093, 0.2628551721572876, 0.24535053968429565], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_fa31f8eb3f6750e8aa15e375f1c859d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b860cca7d74e41392aecd7a90ee59238
    def get_inputs(self):
        return [
            paddle.to_tensor([0.524825930595398, 0.556461751461029, 0.5478492975234985, 0.5832623243331909, 0.5003096461296082, 0.5420165657997131, 0.5428001284599304, 0.555755615234375, 0.496071457862854, 0.5120435953140259, 0.5387029051780701, 0.526077389717102, 0.5147756338119507, 0.48542165756225586, 0.5421690940856934, 0.5022813677787781, 0.49622341990470886, 0.5192069411277771, 0.5390492677688599, 0.5366702079772949], dtype='float32').reshape([20]),
            paddle.to_tensor([0.33159080147743225, 0.4622668921947479, 0.41613614559173584, 0.17420028150081635, 0.32067862153053284, 0.44250139594078064, 0.08052463084459305, 0.4976094663143158, 0.09789836406707764, 0.007808682508766651, 0.19774028658866882, 0.30942657589912415, 0.24886302649974823, 0.29637670516967773, 0.30167829990386963, 0.3173513412475586, 0.41125351190567017, 0.39854443073272705, 0.13736076653003693, 0.026078952476382256], dtype='float32').reshape([20]),
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


class TestPrimitiveOp_2babd070c72583021f1a17211a7abb7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06afa0ac8681719902f3e86d0feb1b9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.8082316517829895]], [[0.8710860013961792]], [[0.8783239722251892]], [[0.8954198956489563]], [[0.8441342711448669]], [[0.8530685901641846]], [[0.9138086438179016]], [[0.883136510848999]], [[0.9044186472892761]], [[0.8764577507972717]], [[0.9035487174987793]], [[0.8104052543640137]], [[0.8776697516441345]], [[0.9435001611709595]], [[0.7308052778244019]], [[0.8622450828552246]], [[0.9171043038368225]], [[0.9255258440971375]], [[0.8856898546218872]], [[0.872915506362915]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_469236e31be1b89a910c08a56a66e0ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2d80b0a464dd4db39b3fdc7fe062c41
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0072457194328308105], [0.12277762591838837], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_fb0b256101f39538982d7472787eefdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2d80b0a464dd4db39b3fdc7fe062c41
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16607365012168884], [0.3619728982448578], [-0.0035280585289001465], [0.2037476897239685]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4147489666938782], [-0.09613686800003052], [-0.3273961842060089], [-0.15094348788261414]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_81b8bb1bab350ca4da4592c835cf793a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2d80b0a464dd4db39b3fdc7fe062c41
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0072457194328308105], [0.14406728744506836], [-0.01705726981163025], [0.12984122335910797]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.38327330350875854], [0.04807387292385101], [0.12588948011398315], [-0.3918229043483734]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_92e66cdf5d2366354d0c3710b6c735f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2d80b0a464dd4db39b3fdc7fe062c41
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16607365012168884], [0.3832625448703766], [0.3038460612297058], [0.3782694339752197]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4147489666938782], [0.15025699138641357], [0.12588948011398315], [-0.15094348788261414]], dtype='float32').reshape([4, 1]),
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


class PrimitiveOp_101fbd0e04ef06ff5e7ab12848c8d49c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2054, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2054, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_979dc3535731acf45a8e7905e8354a51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_101fbd0e04ef06ff5e7ab12848c8d49c
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_979dc3535731acf45a8e7905e8354a51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_101fbd0e04ef06ff5e7ab12848c8d49c
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_979dc3535731acf45a8e7905e8354a51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_101fbd0e04ef06ff5e7ab12848c8d49c
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_979dc3535731acf45a8e7905e8354a51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_101fbd0e04ef06ff5e7ab12848c8d49c
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_979dc3535731acf45a8e7905e8354a51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_101fbd0e04ef06ff5e7ab12848c8d49c
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3bf9482d51855f848dd1192d84f82797(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2054, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2054, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_324e7623deed099947f7f68efc5682f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf9482d51855f848dd1192d84f82797
    def get_inputs(self):
        return [
            paddle.uniform([2054, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_324e7623deed099947f7f68efc5682f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf9482d51855f848dd1192d84f82797
    def get_inputs(self):
        return [
            paddle.uniform([2054, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_979dc3535731acf45a8e7905e8354a51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_101fbd0e04ef06ff5e7ab12848c8d49c
    def get_inputs(self):
        return [
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2054, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_c57bf6782fdb7c6840c035ccea0bad7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0399b6b5890bd2e28b6c4105967fdbad
    def get_inputs(self):
        return [
            paddle.to_tensor([0.354013055562973], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_225e719087b17d3d9046fb3921550b24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5aad34d7c86157d1487993f25703db5
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4633258879184723], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_225e719087b17d3d9046fb3921550b24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5aad34d7c86157d1487993f25703db5
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4633258879184723], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_6069ff8efde42a5874a218b29ba27c56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e2e2d1970fd88b18fd9d1b4d7b374ba
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2569820284843445], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_9b6a51151e0f1d63302c429b39ff0265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3a875073a11ba5ab64d9b4598a017dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 38, 58], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.19514723122119904], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_1ad46da499a5805d25e1aff80645af3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d651b3382091c8333b6aeb3076a83d6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4844519793987274], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_0db6fe764ea8b679f303bb14c1cbd017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c85f2f8d2852001006ac18e16d09892e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 13, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.02733994647860527], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_8d4f2f00ce25d10f17b3d65a47e76525(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bf619cdf78e852feb559b5e72a5ba77
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.07406904548406601], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_e2bb630ae64fde064525c8de4a84f77e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d93c0a755c10350fd60d0534979829b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.24221260845661163], dtype='float32').reshape([1]),
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


class PrimitiveOp_d6916080e1832ec6c357ff3839f376c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4218, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[4218, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d3bee251a52953037fef05095cce7db7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6916080e1832ec6c357ff3839f376c4
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3bee251a52953037fef05095cce7db7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6916080e1832ec6c357ff3839f376c4
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3bee251a52953037fef05095cce7db7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6916080e1832ec6c357ff3839f376c4
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3bee251a52953037fef05095cce7db7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6916080e1832ec6c357ff3839f376c4
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3bee251a52953037fef05095cce7db7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6916080e1832ec6c357ff3839f376c4
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8766e9cfa4f0deea7ee1f8a55d3612c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4218, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[4218, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3bfc61583af640498cdc27646992fa35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8766e9cfa4f0deea7ee1f8a55d3612c1
    def get_inputs(self):
        return [
            paddle.uniform([4218, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bfc61583af640498cdc27646992fa35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8766e9cfa4f0deea7ee1f8a55d3612c1
    def get_inputs(self):
        return [
            paddle.uniform([4218, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3bee251a52953037fef05095cce7db7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6916080e1832ec6c357ff3839f376c4
    def get_inputs(self):
        return [
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4218, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_f8ef5ee9fc5f90fff212339832677f30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_966012030699764a7cc0a9c8e8a699a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 19, 29], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.11168647557497025], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_cabe78ad8dccb9b618c691819a563462(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3890b2dd49a51fa89038161d787d38
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.40316203236579895], dtype='float32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()