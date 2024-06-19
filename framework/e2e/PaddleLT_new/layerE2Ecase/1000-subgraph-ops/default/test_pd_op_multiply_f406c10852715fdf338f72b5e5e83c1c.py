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



class PrimitiveOp_88e51f2bae392ed7d4c6b58929ef6a83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 576, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_84ff80f8752e2e57edd1b5c325eb57ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88e51f2bae392ed7d4c6b58929ef6a83
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_bba7aec2e29072e67649ba96732e6783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f714bbf021d42dc1978a94f2e6726786
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_f761d5d0d1c6822a27a090d13adb6f4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b0bf5dc6a91df258aef9a721530f24e
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.18082675337791443], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_23efb7705a00d013dde414a35a50893b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_990786bb38024e86bf09b9526a4019cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23efb7705a00d013dde414a35a50893b
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_a00003f4653c43f7b87e1666ebb669ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_823e0a5156c49838cbf9d6d06d9c53b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 136, 136], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ba5cc92704ec940eeadc35b0cfa3e2ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b8407e2d8d2f2d788039899ada63862c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba5cc92704ec940eeadc35b0cfa3e2ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_52998a389ca40931146f4fe041e9b7eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3ad5d76ed43394f1063c579701c619b
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 18, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_a5c3c3c06a0d13b96cdf431c9ceb62f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ecfa147eb81fe4e6c4ac0a05c00bd6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a5c3c3c06a0d13b96cdf431c9ceb62f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ecfa147eb81fe4e6c4ac0a05c00bd6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a5c3c3c06a0d13b96cdf431c9ceb62f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ecfa147eb81fe4e6c4ac0a05c00bd6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a5c3c3c06a0d13b96cdf431c9ceb62f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ecfa147eb81fe4e6c4ac0a05c00bd6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a5c3c3c06a0d13b96cdf431c9ceb62f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ecfa147eb81fe4e6c4ac0a05c00bd6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c514e2a5c05418dc0e1b53b10e40f6b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3086425fc951090c34f0bf80c1f10896(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c514e2a5c05418dc0e1b53b10e40f6b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_0aa7dc18fce1e661e33eca43bade9fc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a1b927eb195547e01b6d7d96990efc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_f86c60bc0dc24b78cb4f53b504258cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_860a59f28792c56fa166071671e35112
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_daa08d99651a898b13b0476c62a73d33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21640bfeddd1cc679a8a6fae429cb547
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.3884550631046295, -0.467956006526947]], [[0.01898980140686035, -0.4390466809272766]], [[0.5626223087310791, 0.056461453437805176]], [[0.3931136429309845, 0.12547802925109863]], [[-0.7569209337234497, 0.668235182762146]], [[0.11759907007217407, -0.006632208824157715]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_f4fae92a9425bd9f12f8d3ea342fbe1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21640bfeddd1cc679a8a6fae429cb547
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.21056562662124634, -0.6945126056671143]], [[-0.36513787508010864, -0.6188708543777466]], [[0.45076581835746765, -0.6415443420410156]], [[-0.26054754853248596, 0.40958911180496216]], [[-0.322548508644104, 0.5636998414993286]], [[0.6659334301948547, -0.2623436450958252]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class TestPrimitiveOp_c533f1d5640651781213673a61a71856(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4992024e84675fd31e5997a5c142f74e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.3884550631046295, -0.467956006526947]], [[0.01898980140686035, -0.4390466809272766]], [[0.5626223087310791, 0.056461453437805176]], [[0.3931136429309845, 0.12547802925109863]], [[-0.7569209337234497, 0.668235182762146]], [[0.11759907007217407, -0.006632208824157715]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.3884550631046295, -0.467956006526947]], [[0.01898980140686035, -0.4390466809272766]], [[0.5626223087310791, 0.056461453437805176]], [[0.3931136429309845, 0.12547802925109863]], [[-0.7569209337234497, 0.668235182762146]], [[0.11759907007217407, -0.006632208824157715]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


class TestPrimitiveOp_ba18a88083049dd9d9c9cb4380a153d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4992024e84675fd31e5997a5c142f74e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.21056562662124634, -0.6945126056671143]], [[-0.36513787508010864, -0.6188708543777466]], [[0.45076581835746765, -0.6415443420410156]], [[-0.26054754853248596, 0.40958911180496216]], [[-0.322548508644104, 0.5636998414993286]], [[0.6659334301948547, -0.2623436450958252]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.21056562662124634, -0.6945126056671143]], [[-0.36513787508010864, -0.6188708543777466]], [[0.45076581835746765, -0.6415443420410156]], [[-0.26054754853248596, 0.40958911180496216]], [[-0.322548508644104, 0.5636998414993286]], [[0.6659334301948547, -0.2623436450958252]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class TestPrimitiveOp_6a28ce710de91b87d43baf2003a0d16b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe310264e1b88b3efe1ed272b8463698
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.22495286166667938], [0.08486909419298172], [0.18079178035259247], [0.0702679380774498], [1.0293428897857666], [0.001634106389246881]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([[[-0.41431671380996704], [0.18809157609939575], [-0.0690925121307373], [0.07443028688430786], [-0.4739932119846344], [0.045438289642333984]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_c298d8a81c70b0d721544b82a717ec5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe310264e1b88b3efe1ed272b8463698
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3822321891784668], [0.3710111975669861], [0.482023149728775], [0.11439217627048492], [0.2739379405975342], [0.36667025089263916]]], dtype='float32').reshape([1, 6, 1]),
            paddle.to_tensor([[[-0.41431671380996704], [0.18809157609939575], [-0.0690925121307373], [0.07443028688430786], [-0.4739932119846344], [0.045438289642333984]]], dtype='float32').reshape([1, 6, 1]),
        ]


class PrimitiveOp_e3f803a756869e501116ea42698960ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 100, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 100, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_64dd70942b2bc701cfbe4938d1fe843d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3f803a756869e501116ea42698960ea
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_057715d92dae638dcf6709c1155905fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 100, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 100, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1855833d1a5d0b933bfb611165712f81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_057715d92dae638dcf6709c1155905fc
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b8bcd5775fc0db4d50eb321309514e9e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 100, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57bee0f8676a7a7f95e587c0f67f4309(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8bcd5775fc0db4d50eb321309514e9e
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_3e7d5615e7691837695aec5195da788f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d8d81b646705ca1ac48d7efa39592c8
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3e7d5615e7691837695aec5195da788f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d8d81b646705ca1ac48d7efa39592c8
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([100, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b1571de3b7504056e385f67f78d5a62f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f3d932ca3fab708661f9d8a4131a18f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1571de3b7504056e385f67f78d5a62f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_3974a7de256bc2b652e8ac7a11dfe712(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5676c8cad1ef2749a80be2440823e881
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 22, 22], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_990786bb38024e86bf09b9526a4019cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23efb7705a00d013dde414a35a50893b
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_27cfe7bbb53e48150e7fc2e5e9e9f2f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b035ba371ec230244f260b4bb69367b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_abb74cdd85236b7a04fef0f881e0aa36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdbb53a9fcb2ba0cbe12e542d72858bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 76, 116], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.16901332139968872], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b8407e2d8d2f2d788039899ada63862c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba5cc92704ec940eeadc35b0cfa3e2ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_b9cad8bc73fe3be91b8066dce5a203e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_326670e887bb48ce03d78c1dbd808ed3
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 44, 44], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d5d7fb6f0376c7b9f25196718fa7bf2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1025, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_516169286ca45b241be0fff3598d1008(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5d7fb6f0376c7b9f25196718fa7bf2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1025, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c84cf18a00fe0b4231e2a402c952a07a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1025, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b93eaca5afeb811c87121021cc24b3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c84cf18a00fe0b4231e2a402c952a07a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1025, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_488951730b49cdfc732a85c477f73e76(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e9b544ab6f86535ea298ae4199ef4fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_488951730b49cdfc732a85c477f73e76
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_d0477ea01189f8bdd95bd22b4a41d786(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_304b82d9e423ebd888952055c9992f78
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_7206a58f51f8e056f60a538c9063d14d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5e00ddb5250321479259e986cdd2528
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c6ca159074f8fe14958b4df4adf001e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 400, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fdb885db47ca7519ffce04f41009849b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ca159074f8fe14958b4df4adf001e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_0e669ca199540645571042acac10d294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c57e5b50d1f9ddf8c6a107db66b6f238
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_41503b916037a1114b1acca6fdba4bda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7e8e9f2153719136b8e0a20b8cde799
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_41503b916037a1114b1acca6fdba4bda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7e8e9f2153719136b8e0a20b8cde799
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_41503b916037a1114b1acca6fdba4bda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7e8e9f2153719136b8e0a20b8cde799
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bf6ae6ee44a92275d56a63ee17bb8b8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cd881e12589c179bea1331142d21d6e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf6ae6ee44a92275d56a63ee17bb8b8e
    def get_inputs(self):
        return [
            paddle.to_tensor([1508], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
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


class TestPrimitiveOp_28215bcd0078971b871cfb492ea99351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_eea4303ddd61e3133de756e7595fa3f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eea4303ddd61e3133de756e7595fa3f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eea4303ddd61e3133de756e7595fa3f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eea4303ddd61e3133de756e7595fa3f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_10c31942c3373c35133733a54f925542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1508, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10c31942c3373c35133733a54f925542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1508, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eea4303ddd61e3133de756e7595fa3f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1508, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_71cb157ab0aca71badca36e76749a31f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_752a8f3f78ec45b5d0d71fb92718baa5
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 16, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9d4597085202c1282edccf247249bbb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf6ae6ee44a92275d56a63ee17bb8b8e
    def get_inputs(self):
        return [
            paddle.to_tensor([2377], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_415e2b7fe5d61bb3c931eb1db8ca135a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_044a3ce606c087ea7268627004a58797(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_044a3ce606c087ea7268627004a58797(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_044a3ce606c087ea7268627004a58797(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_044a3ce606c087ea7268627004a58797(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_57645385ad3e0ec3ea61f17f63f1c2ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([2377, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_57645385ad3e0ec3ea61f17f63f1c2ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([2377, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_044a3ce606c087ea7268627004a58797(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2377, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f3d932ca3fab708661f9d8a4131a18f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1571de3b7504056e385f67f78d5a62f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_b77d8839a4e10d17246f9d632266806c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6082a6401640964013f8cd8561b907ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 17, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7541fd6af8509f0ce8734aa60e436973(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d45ba9b26809ee86adf3dadcc6e03895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7541fd6af8509f0ce8734aa60e436973
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_a68ad5d6d32ec44453374e15529dd9a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eba1aa160d29bbb44511b24b54850e0
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4cb3ba4434c58aa80180638599ef5e1e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1024, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07ef80ef11a12fcc41f972f3268d0a33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cb3ba4434c58aa80180638599ef5e1e
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1024, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c4285dddd0ee1652a8109bdb8252a752(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1024, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_725f07ec034343c4a0239218df0c20bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4285dddd0ee1652a8109bdb8252a752
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1024, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8ed8e4adce7212e5b3a38ddfc02d28e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_42040ac35d296a5dd9f54ff410604f31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ed8e4adce7212e5b3a38ddfc02d28e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_67724288b6973cda590988c2f80de015(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 2304, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 2304, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1ad5af2a5e18b06020eecc06e2018aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67724288b6973cda590988c2f80de015
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 2304, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_641c50dfc544fbaf7ba31caaf32f7184(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 2304, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 2304, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1d0f9992b1fe7f3fde432d8242de3e2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_641c50dfc544fbaf7ba31caaf32f7184
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 2304, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_825c81def78e3bd13e0f78fa5b389260(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 2304, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1d88d02f17bb384602976dac336981a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_825c81def78e3bd13e0f78fa5b389260
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_dec0647c71c31d41bc4b5798acbefac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57d04499dbe9fd9080bb3e8010b53874
    def get_inputs(self):
        return [
            paddle.uniform([171, 480, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8407e2d8d2f2d788039899ada63862c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba5cc92704ec940eeadc35b0cfa3e2ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_f030d55200ba4a60cdca5f2ced9cd045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740ad0093b4ec2fa38d08f27bbe1086f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ce361428574103724a95686f3ca8862b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 3136, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 3136, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c33240ccb715a70d663d4e013486963a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce361428574103724a95686f3ca8862b
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7d1d45e84957895b7ac2b031636b2994(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 3136, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 3136, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_87de9916d58a93a2e8309db1bcd73865(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d1d45e84957895b7ac2b031636b2994
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5de1e7f314ab59f7cc77bd918188edb3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 3136, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_386839899c9f9a689154df9cfaa7e532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de1e7f314ab59f7cc77bd918188edb3
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6dd8f5115e9b9b191e1a05315edd9c17(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df3f0598f249e5ca08f920ce6d147745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dd8f5115e9b9b191e1a05315edd9c17
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_71e2a12aa137fcb53ce7588a6bad1a1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b5bfa76040afbc7d0b8f79979650f5c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 120, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_05b9bb8eae4016c3fd8b4548ed85dab6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 3136, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 3136, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8fc57001711b806a6526ca853ea32bd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05b9bb8eae4016c3fd8b4548ed85dab6
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9c45914fe8b40a284028e09cbdb673ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 3136, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 3136, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_790b39e3e89f7e03563138cd27d97ce8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c45914fe8b40a284028e09cbdb673ee
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f1677f2290efe9a742560ef7d7dfb3ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 3136, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_be67299aa43f002268392a6c49d469ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1677f2290efe9a742560ef7d7dfb3ba
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_3d28124ef9da8b74d34e51fa89c120f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d07132c8b6e809630ba97c8e007c0d37
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f6e35b8f487625603ffedcfab9df3688(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 196, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b92b13d27e3aa19ce00b8d592d5cb336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6e35b8f487625603ffedcfab9df3688
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9b2456f79ae510f435d8c854742cf134(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 196, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a748b009e681bf2b8308e75a7caa464(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b2456f79ae510f435d8c854742cf134
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b01629033fc9528fcc55a5bfc15c4fc2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_17696e5af717e1533cedbc1d01606ee9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b01629033fc9528fcc55a5bfc15c4fc2
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_489733ffc3f85b8db860a8ef24c1f752(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_484bfd9d6d165443f303a2927a245e83
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7996d0f16b68a00175ac2ee58b32986a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e0242c1fa2029637fa51c2ec6c7c9569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7996d0f16b68a00175ac2ee58b32986a
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_9ca878e9e42a68cd3870dab42ed518a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efec768d4ce4f13c30448ea8fb310dbe
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_92eab07f30958316a00a156a8941a9ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d93c0a755c10350fd60d0534979829b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.10206487774848938], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_d98d6e25dc31fe929b92b070c52a94fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_966012030699764a7cc0a9c8e8a699a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 19, 29], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.4136306047439575], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_3a5c0916297b8397cb48b484b07f7ef9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32074b59bac5abaf1b8a9d10910ce304
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a5c0916297b8397cb48b484b07f7ef9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32074b59bac5abaf1b8a9d10910ce304
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a5c0916297b8397cb48b484b07f7ef9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32074b59bac5abaf1b8a9d10910ce304
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a5c0916297b8397cb48b484b07f7ef9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32074b59bac5abaf1b8a9d10910ce304
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c19d801d8a9adac0c877ec38bfa79311(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4fe93ab0db01dd2a2aaec1c7bbe3ddcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c19d801d8a9adac0c877ec38bfa79311
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_45888e0c3a938ba413a08e089195ccf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_674a820d16d84b0c122083859b96efc3
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 44, 44], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_a2044eee0eb8e9022ddd30213fd26f48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aed20c7e958de824a596996be23cd1b
    def get_inputs(self):
        return [
            paddle.uniform([145, 60, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7a1bf0d27f23bea756c72d3877ffa12c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 197, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[54, 197, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7f2f7c7cedf4fd5e421198c7052f7321(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a1bf0d27f23bea756c72d3877ffa12c
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 197, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0e5f5e6415c449229d084cca04f33f61(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 197, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[54, 197, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74b2d7031083edd068e37e644171f2ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e5f5e6415c449229d084cca04f33f61
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 197, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_85065914e0f9ba03e3c997f8b95d3f2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 197, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_110af96d89c9362d4ab53083c03a1464(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85065914e0f9ba03e3c997f8b95d3f2b
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f0f893a89be09dd2f9000d61202b6d7f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 784, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 784, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_eab687913e0e3d772d31dbf05424e461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0f893a89be09dd2f9000d61202b6d7f
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_12df68d89b8813493227a5d5192e489b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 784, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 784, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e19dee740eca47ae59647d3ffe4a8d0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12df68d89b8813493227a5d5192e489b
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_05eb97b286a792a80f7726c6363886ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 784, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_10e194c8bac5598853dea89b168480b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05eb97b286a792a80f7726c6363886ec
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_e8f28e8d97e68996e9f085968b2968d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2d80b0a464dd4db39b3fdc7fe062c41
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3121303915977478], [0.0], [0.0], [0.3540444076061249]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.1762411892414093], [0.0], [0.4803326427936554], [0.0]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_6b8cc0b3e12e7b2b510c372a29639746(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2d80b0a464dd4db39b3fdc7fe062c41
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.8465824723243713], [0.29671111702919006], [0.7947391867637634], [0.3643512427806854]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.1762411892414093], [-0.10018911957740784], [0.5332373380661011], [-0.5061129927635193]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_7655fab9d74184fb3237b21b307a8929(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2d80b0a464dd4db39b3fdc7fe062c41
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3121303915977478], [-0.5012533068656921], [-0.10262379050254822], [0.41398754715919495]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.6741307377815247], [0.39302515983581543], [0.8384606242179871], [-0.9249767661094666]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_a041aa6f7c0a61116c19d5570489b325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2d80b0a464dd4db39b3fdc7fe062c41
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.8465824723243713], [0.29671111702919006], [0.7947391867637634], [0.4242943823337555]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.6741307377815247], [0.39302515983581543], [0.8913652896881104], [-0.5061129927635193]], dtype='float32').reshape([4, 1]),
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


class TestPrimitiveOp_cf5c555dcd8587a7e4e767592de80282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec38d62e74c14a10210d7629afce5c6a
    def get_inputs(self):
        return [
            paddle.uniform([43, 80, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_da15b806a108e439ec344bf4875cfb31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c49b4c9d4c069b5e0893f4249f7b850
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([8816], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7c97cea5ab0e5517d2625e1cac8042a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5478c05900365ed4b8a3d3e2d3ecceca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c97cea5ab0e5517d2625e1cac8042a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_6c8c49097f793bb4815d54c6395210e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e23867604478c3bdd9e77d3c65cfb38
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_79fab2ddab9377dd34e70e31aa78037e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_42b764715d2d3475c83e764c32d049b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79fab2ddab9377dd34e70e31aa78037e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_f78cdb96605faa717c6732b28e0dff03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bf64d7cf8054186bcc7796217057d48
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_df3f0598f249e5ca08f920ce6d147745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dd8f5115e9b9b191e1a05315edd9c17
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_7601b8e0e4549d22c0d5394e5131e1c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_107d3bbe2a770bb1bd427861fb29d15d
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_2a4779f7e62a26ae3e4dbed3c1450fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bda26e4a9a5300d0796dce55d689a4
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_32ae02440098c63baf59fb574aae1c13(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a9148ad62922d4d07437296be051718f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32ae02440098c63baf59fb574aae1c13
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f86c60bc0dc24b78cb4f53b504258cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_860a59f28792c56fa166071671e35112
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_df3f0598f249e5ca08f920ce6d147745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dd8f5115e9b9b191e1a05315edd9c17
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_954e26c8469a66b0415a257fcc089d33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef10b1b63154d158c6d96593a58b1f5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 88, 88], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_73928626126134a70fa7d944149921a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.5967594385147095, -0.57695072889328, -0.5512745380401611, -0.4258155822753906, -0.7427231073379517, -0.04512202739715576], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.18019214272499084, 0.0, 0.13494634628295898, -0.46017640829086304, -0.5001617074012756, -0.581457257270813], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_825b6f968a35068903cb3356ee7c9c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1075313612818718, -0.0, -0.07439248263835907, 0.19595028460025787, 0.37148165702819824, 0.026236530393362045], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_54ec0ec489373b4066dcd78dc45fffa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, -0.0, -0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_b425c369121a925423de4e34e72bb892(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 0.5090863704681396, 0.0, 0.0, 0.0, 0.11437070369720459], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2738617956638336, 0.0, 0.6923106908798218, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_58db60e82eeee7fdd8bac34ff5a1a4d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.23735523223876953, -0.57695072889328, -0.5512745380401611, -0.25409436225891113, -0.22113677859306335, 0.03664344549179077], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.16398099064826965, 0.5595255494117737, 0.13494634628295898, 0.19261020421981812, -0.03280341625213623, -0.33978235721588135], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_a0de99def1b4e250ae348fa34ad21817(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3846908211708069, -0.06986401975154877, -0.13380378484725952, 0.10281282663345337, 0.5327516794204712, -0.12062910199165344], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3846908211708069, -0.06986401975154877, -0.13380378484725952, 0.10281282663345337, 0.5327516794204712, -0.12062910199165344], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_32d3893e58f7cc61dc62d74a5f6f6a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.23513256013393402, 0.20751801133155823, -0.018665790557861328, 0.2762005925178528, 0.32108187675476074, 0.13652008771896362], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.23513256013393402, 0.20751801133155823, -0.018665790557861328, 0.2762005925178528, 0.32108187675476074, 0.13652008771896362], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_5ffc5bc17ac0d65cf2e5d1f6d084c25e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.35940420627593994, 0.5090863704681396, 0.0, 0.1717212200164795, 0.5215863585472107, 0.19613617658615112], dtype='float32').reshape([6]),
            paddle.to_tensor([0.35940420627593994, 0.5090863704681396, 0.0, 0.1717212200164795, 0.5215863585472107, 0.19613617658615112], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_dd2a15568b801efba44f4113959da7c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2900729477405548, 0.5595255494117737, 0.6923106908798218, 0.6527866125106812, 0.4673582911491394, 0.24167487025260925], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2900729477405548, 0.5595255494117737, 0.6923106908798218, 0.6527866125106812, 0.4673582911491394, 0.24167487025260925], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_f073b3d9a9c50a501cacb1636b25e833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.6341857314109802, 0.30571651458740234, -0.3275994062423706, -0.6210290193557739, 0.35463717579841614, 0.03954076021909714], dtype='float32').reshape([6]),
            paddle.to_tensor([1.5647895336151123, 0.754324734210968, -0.8083186149597168, -1.5323266983032227, 0.875031590461731, 0.09756284952163696], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_3c77a3981058f591b81ec6345ec9fc98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4864a9a1fb18d135421bcdec4268d3c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.49808448553085327, 0.18739454448223114, 0.20936408638954163, 0.4876049757003784, 0.23682691156864166, 0.0038428844418376684], dtype='float32').reshape([6]),
            paddle.to_tensor([0.9923672080039978, 0.23060952126979828, 0.2648046910762787, 0.9516193270683289, 0.3103187382221222, 0.0038577092345803976], dtype='float32').reshape([6]),
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


class TestPrimitiveOp_87f808908be10b63fffb588a3ed89626(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3da1e67cebce87064c8d2b0a9e3fef2
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 1, 49, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f7fb8394b9cfd76a24b18ef5b1d9966e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 320, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d45385c53786eecf5b1eda25f491b5fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7fb8394b9cfd76a24b18ef5b1d9966e
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_56c0ed305d2e41e30fb08d708f806d7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a349844fda90726045bdef2e5166888
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 9, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_1e805788869cd3c6c3f73e97177edac0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ff51b96b9086fc098740f35a266b6e9
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_78d431f9249bc01bb3f301a8499b5e32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f82bae9f01017bc84c73b0afb0ceb402
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.05925503373146057], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_b592a56e0cc44b65b004b1247796f2f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a362ef335a69f09b07b96404efdcbbd8
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5328717a8cee22cfcdd245b743fb930d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80af52dfe292d72107a04668b0cd1343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5328717a8cee22cfcdd245b743fb930d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_31f3d888ca69b68bb9233c98c162bcc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_020f692312cb5a1a1f46a6a93b1b8895
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 68, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a11a838d9bfb17d4b04b2b87572d5082(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 336, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b0f82181b3adf2561c3103da1fb6335b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a11a838d9bfb17d4b04b2b87572d5082
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_590c5847ad3fce30103301a75d422c72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45337722230d20739d122b436c01825f
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_604ffdd8a2325ef4c6daaf65dd515e55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5b8478af74dca2c2504ea117ed35028
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 1, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9505370c5739e6fedae3977e111b6faf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_15bb1314c0dfd729d6b913b2b6ef6773(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9505370c5739e6fedae3977e111b6faf
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_60a34fa131d38bd74b1985da6eb9b8e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64b3661e08a9412b02da4daf9ca243b1
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 1, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_d8d474a6109712b807070ac7fc376a9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b20b0336e12a00d9214d414a53a93361
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15200], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2481e105c4881796aef69e0dd25a9d25(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_36363d7e097a47568edd8771d4a9d17a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2481e105c4881796aef69e0dd25a9d25
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_41fa8d43a72cd18d9d5ca4076949039d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd79497871969a862d4fbceea36d0efa
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_990786bb38024e86bf09b9526a4019cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23efb7705a00d013dde414a35a50893b
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_1ed2ea9757bfb9e2ce8722764f82b075(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3be3c6893175a59793a356157187c79
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d79e6833d3f91e08207755f2dfeb43d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5328717a8cee22cfcdd245b743fb930d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_7d79f42dcfbdf5c90d78837d6116738b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92cb17370c1018d4ec03807ee3b617dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 168, 168], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_52f07009d4706a34e98c4fc538573a7e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 49, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e8d97f7bbcba30ab4488e8dd40165c4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52f07009d4706a34e98c4fc538573a7e
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a65a88e593508bcd4e1535187164de07(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 49, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b33be8029cb089241fbcbcf6c2b159bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a65a88e593508bcd4e1535187164de07
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9cc6595815faeb63a4a021450164ab30(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3baefed6591e7b4a852ea642bb4806be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9cc6595815faeb63a4a021450164ab30
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d45385c53786eecf5b1eda25f491b5fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7fb8394b9cfd76a24b18ef5b1d9966e
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_a05bee14335316b5c2e87e8bd513a094(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1f1df3749cd6f1f3a332144db4ce2aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 22, 22], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_cdb6604d56aa3690449a420f12aa9962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b34b55aabf7993d26444c37fd95b6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f3d932ca3fab708661f9d8a4131a18f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1571de3b7504056e385f67f78d5a62f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_34ba4683726bfa060412ef9a187e3173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90bdd5f80aeb40d33d559c31d86fc3f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4359d6bbd4c846678c76b079139d587c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e7f0cbdd84b5f6098234910a7cafb5d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4359d6bbd4c846678c76b079139d587c
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_b2708fa5f4a470d14b8940ff27a34aae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37d431d8d12ac49415bb9099f8f1b721
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ab75d6f916b8c515969c8108422557d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 49, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c8cd99e444dad3dc78e88ca0a51adf00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab75d6f916b8c515969c8108422557d3
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_597a6361be11b23fe2d262c45726cc7a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 49, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6172caada93e27ce0f496ad3beace0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_597a6361be11b23fe2d262c45726cc7a
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c54270e10cb3f0d4753cec8cd8b9ecda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_27616106a6dd86d5554fe1a1875bd07d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c54270e10cb3f0d4753cec8cd8b9ecda
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_382b20b4b9f61800ddbe43e1cf3aa712(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fc219cf526e458de33df21ff9de38ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382b20b4b9f61800ddbe43e1cf3aa712
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_949cc732c943b4a58d178618728d32be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0250f32c341932761eafea94816444b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 15, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c502fcbe29607f05f11fd85f22d480ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 640, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 640, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5cbd98887fe60648c15f6507e1c81324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c502fcbe29607f05f11fd85f22d480ef
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 640, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bc82976749dcf5326b450840ed243276(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 640, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 640, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f0b20594611e4a89e04a418664c7485b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc82976749dcf5326b450840ed243276
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 640, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6e4d2e60b1e6226404d18516436a1011(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 640, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_feb8a9b3e0e391ef53fdacf0ddc9c644(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e4d2e60b1e6226404d18516436a1011
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fdb885db47ca7519ffce04f41009849b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ca159074f8fe14958b4df4adf001e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_72a5d9b64e8c6b3219ee09efb082aafc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a3d3de657dfd343ed2575898fb6f779
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 9, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_e88cbd8a584a5c33d26a11b2f8d08f1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2472b300c494153848ce36201cd5f902
    def get_inputs(self):
        return [
            paddle.uniform([22, 2, 1, 9, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b592a56e0cc44b65b004b1247796f2f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a362ef335a69f09b07b96404efdcbbd8
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_df3f0598f249e5ca08f920ce6d147745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dd8f5115e9b9b191e1a05315edd9c17
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_7688d7f72aa5cbf6d99723a3abf67778(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f87f6bf733fc6abbb4e44c5c18dd24a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 104, 104], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_4fd09b76da9c622599805faabf65b159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123c4c37f31fa937c21ca370a3424f4b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.11026203632354736], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e8d97f7bbcba30ab4488e8dd40165c4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52f07009d4706a34e98c4fc538573a7e
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b33be8029cb089241fbcbcf6c2b159bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a65a88e593508bcd4e1535187164de07
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3baefed6591e7b4a852ea642bb4806be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9cc6595815faeb63a4a021450164ab30
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_399dfdf943178a054f43f28b7abfd3bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_204265cbd2e15645b84b480dd7a0290c
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_baed70d442803c3e98b03d9699acfab9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b24ad60e80ab24bb6b066b48623adab2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_baed70d442803c3e98b03d9699acfab9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b24ad60e80ab24bb6b066b48623adab2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_baed70d442803c3e98b03d9699acfab9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b24ad60e80ab24bb6b066b48623adab2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_baed70d442803c3e98b03d9699acfab9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b24ad60e80ab24bb6b066b48623adab2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_ddf66a9789bd8156ad7917f06fe30c1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c85f2f8d2852001006ac18e16d09892e
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.3365565538406372], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1e805788869cd3c6c3f73e97177edac0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ff51b96b9086fc098740f35a266b6e9
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_e1e4cf7960f8860e4f589cb4480e9022(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1272b57e0e52d16007171347313276a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e1e4cf7960f8860e4f589cb4480e9022(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1272b57e0e52d16007171347313276a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e1e4cf7960f8860e4f589cb4480e9022(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1272b57e0e52d16007171347313276a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e1e4cf7960f8860e4f589cb4480e9022(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1272b57e0e52d16007171347313276a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e1e4cf7960f8860e4f589cb4480e9022(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1272b57e0e52d16007171347313276a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_84ff80f8752e2e57edd1b5c325eb57ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88e51f2bae392ed7d4c6b58929ef6a83
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_54cf89b888a729615e6e77118b9b7f35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_979512685801d46188aeb0d32fdcdda2
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_424078796bf2c8c4f9830dd4913cfb52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d2c9a26cef048330e50504fc2591152
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_424078796bf2c8c4f9830dd4913cfb52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d2c9a26cef048330e50504fc2591152
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([300, 80], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_609ee1c8af480c718f5506a98092efa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6063ee912be447da30887d64278c98a
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4df27415c72a07f3045dd0694737537a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c19d801d8a9adac0c877ec38bfa79311
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_19061d13f67c0e491858b43a336e3fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd6884c7e30465b66269ab96919ba6bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_bcb9873187d5fe3989abe828731bb5fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4245913426c50e23e26e996c4e3e55d4
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b31fe694663a63c77eaf9110c02771f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79fab2ddab9377dd34e70e31aa78037e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_1e01f6c506aa50b8b76c763840202556(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77edbdf76945be9cef813f9fcc8152a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_b7fa01608f1db5b2c21173b2cd924016(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a44734e17f450d9c0dbb1333fb739ac0
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.3007093369960785], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_81d127e66eeaf3a06d2a06fdf9e5c504(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dd8f5115e9b9b191e1a05315edd9c17
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_e20abf0f93fcd1ebd3caacbc67614978(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6eb0e15a01a84c60f193e0d07796a34b
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8407e2d8d2f2d788039899ada63862c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba5cc92704ec940eeadc35b0cfa3e2ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_4006f8ec9fea4c732a9bb886ba8729eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ff4032347afccea751b9d7652d07ac0
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 88, 88], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_83ed8bc83fffccfcee982bd71f84cdfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d636a839c1554b6e0420f4abb46f545f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83ed8bc83fffccfcee982bd71f84cdfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d636a839c1554b6e0420f4abb46f545f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83ed8bc83fffccfcee982bd71f84cdfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d636a839c1554b6e0420f4abb46f545f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83ed8bc83fffccfcee982bd71f84cdfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d636a839c1554b6e0420f4abb46f545f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c33240ccb715a70d663d4e013486963a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce361428574103724a95686f3ca8862b
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_87de9916d58a93a2e8309db1bcd73865(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d1d45e84957895b7ac2b031636b2994
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_386839899c9f9a689154df9cfaa7e532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de1e7f314ab59f7cc77bd918188edb3
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_080675a235e8c033a12b931358fa5a9f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 200, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 200, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9cb765c657df2c4aeb4731ca4ab7b3ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_080675a235e8c033a12b931358fa5a9f
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 200, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bd357f5d8de2fc456e1142651a516690(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 200, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 200, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_10e66d4ef28f5565ba2fb06eadba4809(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd357f5d8de2fc456e1142651a516690
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 200, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bd9dc1b01f8b93cec463e59081b6c52c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 200, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_784e3f6393cb378e05255f4adb3cd1a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd9dc1b01f8b93cec463e59081b6c52c
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b5af49e0f99aa9f32c14a8800eafca8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 49, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_41ec89f6832fbc97bd427594cbcd5775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5af49e0f99aa9f32c14a8800eafca8d
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5e948c5ec06727a5553c93d6b9d5e514(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 49, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c104dd895cba4f3f20bf661588c3424d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e948c5ec06727a5553c93d6b9d5e514
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6f99381c513c67f933e83be887b62389(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_812ad552a1a75b71c984cc9c81c4aeff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f99381c513c67f933e83be887b62389
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e0242c1fa2029637fa51c2ec6c7c9569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7996d0f16b68a00175ac2ee58b32986a
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_6af6cf0e4f054c03852c15733fa911ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73ac9e1788bcb88b4f8ecb5ae2b2ae40
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_7621207c63cefb9dec7e0aa4a84df6ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b2780f69624fff122ed8c64c70bbb6d
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.07846301794052124], [-0.1516473889350891], [-0.6138649582862854], [-0.12660488486289978], [-0.02549988031387329], [-0.13457483053207397]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.8236229419708252], [-0.7964882254600525], [-0.29898810386657715], [-0.1032944917678833], [0.2434830665588379], [-0.23272746801376343]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_a148b8d7f1360332a010bad79dcdfbf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b2780f69624fff122ed8c64c70bbb6d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.447638601064682], [0.14713245630264282], [0.594458818435669], [-0.48198577761650085], [-0.22798597812652588], [-0.2091558575630188]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.058366745710372925], [-0.6479198932647705], [0.4881892502307892], [-0.08177691698074341], [0.1049279272556305], [0.29351410269737244]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_e7429e3a0d5d0a3c0780b0b89e6b8d1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b2780f69624fff122ed8c64c70bbb6d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.621975302696228], [0.14713245630264282], [0.594458818435669], [-0.05641886591911316], [0.37385785579681396], [0.03538200259208679]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.9234768152236938], [-0.6479198932647705], [0.4881892502307892], [0.08676105737686157], [0.3561587929725647], [0.29351410269737244]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_3086425fc951090c34f0bf80c1f10896(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c514e2a5c05418dc0e1b53b10e40f6b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_05efad70df4abff6c32caa2c24575344(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_005762fbb1c18ab515f0079ae9d762cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e7f0cbdd84b5f6098234910a7cafb5d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4359d6bbd4c846678c76b079139d587c
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b31fe694663a63c77eaf9110c02771f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79fab2ddab9377dd34e70e31aa78037e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_764740876c9354b8207be747c2be9e59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac7b51980a8ac159818613a8b9043cc3
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4df27415c72a07f3045dd0694737537a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c19d801d8a9adac0c877ec38bfa79311
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_edc1da46a0a0212f93392a3ceab061d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e425d127966b14bc1ff56d6858f83625
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_df3f0598f249e5ca08f920ce6d147745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dd8f5115e9b9b191e1a05315edd9c17
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_48772bc6151a57a276f8591379e13fb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81308db06791c5f7841fe9f2d2f68387
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f75c0dd4434232b4fd14b27654a58105(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d725ad75482a80a14e59b3b3c00fc205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f75c0dd4434232b4fd14b27654a58105
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_56f5d0da385f214c8ec917c027c7c5ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c8b58b96a0112d6b478b87a960a84ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4cc472645e01aca6781df184b92f0cfd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 196, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d2c918f076d7d4f82ade29a3686b180d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cc472645e01aca6781df184b92f0cfd
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3a280b542eb2666a0f42ad377c7d05bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 196, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d5e48406f87d8dfe173215c62b3b38d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a280b542eb2666a0f42ad377c7d05bb
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8198545b41671385e47b0db8d12c7556(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5a97e3c13d003fbdf95d3791703211c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8198545b41671385e47b0db8d12c7556
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_ec2442c3da67124857cfd77b695d1e8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_373f28edd813211622ed1dd23f15b0b4
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ec2442c3da67124857cfd77b695d1e8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_373f28edd813211622ed1dd23f15b0b4
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ec2442c3da67124857cfd77b695d1e8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_373f28edd813211622ed1dd23f15b0b4
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_ee0952ffa4beabf359a88c9149d51914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_458ab14b7d947a9d88ede4190462b605
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c9f62a0d6f800f1d2ef24a9da47ca91b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0729d5e983964a104c9f321b6a674da0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9f62a0d6f800f1d2ef24a9da47ca91b
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f3d932ca3fab708661f9d8a4131a18f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1571de3b7504056e385f67f78d5a62f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_964c9627e6af265968d25582e6abcf21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_329d627d1789960dbc7ad0c6299c5604
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_622ca4fce08683dab2491438ceb8c2d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 44, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9a6a36e2e47e9e78e5ca0f5a6a8e3618(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_622ca4fce08683dab2491438ceb8c2d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_5631b0de3318879403794e67dab21706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9ac2ea86a8d7e7133229c18b0519997
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b31fe694663a63c77eaf9110c02771f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79fab2ddab9377dd34e70e31aa78037e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_dda4f832d07039c7485b24ac0b360458(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5acca4d79884cb836a8cf2ff2b46102f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_5fd6f67483ede6dd9ee7554d9d78c321(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90997f531f3da2136342d353b81fee34
    def get_inputs(self):
        return [
            paddle.to_tensor([2.333179473876953, 2.362571954727173, 2.6401143074035645, 2.073599338531494], dtype='float32').reshape([4]),
            paddle.to_tensor([0.797734260559082, 1.190996527671814, 0.6138838529586792, 1.4411441087722778], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_6bed02a4da1373162e6cfc8b6db339be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90997f531f3da2136342d353b81fee34
    def get_inputs(self):
        return [
            paddle.to_tensor([2.025268077850342, 2.240166664123535, 2.293292999267578, 2.3765785694122314], dtype='float32').reshape([4]),
            paddle.to_tensor([0.20226573944091797, -0.19099652767181396, 0.3861161470413208, -0.4411441385746002], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_48f8888dfc37a1f9964120a5f725d464(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90997f531f3da2136342d353b81fee34
    def get_inputs(self):
        return [
            paddle.to_tensor([0.567724883556366, 0.5964877009391785, 0.6265502572059631, 0.4849854111671448], dtype='float32').reshape([4]),
            paddle.to_tensor([0.06841081380844116, 0.39981400966644287, 0.2697553038597107, -0.3491121530532837], dtype='float32').reshape([4]),
        ]


class TestPrimitiveOp_e0242c1fa2029637fa51c2ec6c7c9569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7996d0f16b68a00175ac2ee58b32986a
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_6dfd0ba9806ae18c978ab1e9d647c88f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23d632587381744b828c29127c0434d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_194b11587c47088f1c2131d4d4ac0ce9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 56, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_655fae65f2acc98c547a837422c0b500(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_194b11587c47088f1c2131d4d4ac0ce9
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_c979e668057f1666e50c51a5112fb14c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99399950c34280c4c605c0aadc4c35f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_2c9f5ff6ce317d355ada8b5daaf1ba33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b73850f75e1007e6dddd8446f9d1329e
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2c9f5ff6ce317d355ada8b5daaf1ba33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b73850f75e1007e6dddd8446f9d1329e
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ca77fa8049ffecf3ee5f3b6dc995a0b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 21824, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07e17b91dc60b2cf3503a5165c731533(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca77fa8049ffecf3ee5f3b6dc995a0b1
    def get_inputs(self):
        return [
            paddle.to_tensor(-1.0, dtype='float32').reshape([]),
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2c9f5ff6ce317d355ada8b5daaf1ba33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b73850f75e1007e6dddd8446f9d1329e
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_68e96d9ec0d1c437603475b6006b867d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 49, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c30355eee7f2640258da76929d2f3298(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68e96d9ec0d1c437603475b6006b867d
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bf5e69266b9c161f750432b2cda26b49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 49, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c06901c15c1248c6a176e101b19fc431(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf5e69266b9c161f750432b2cda26b49
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2dad5f95ffd3ba554f6eeaa58d437952(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_efc6a9c94b8ce79c9be986013405e141(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2dad5f95ffd3ba554f6eeaa58d437952
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3086425fc951090c34f0bf80c1f10896(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c514e2a5c05418dc0e1b53b10e40f6b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_f83d4e7655c90c076acf68db5735ee1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11fa09a6b5c44678748f93ef61cb709b
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 11, 11], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_378965dacbacd533acfaeb8dc8c6e076(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a36ba04b41c7c843bedbb4d423b2a09
    def get_inputs(self):
        return [
            paddle.uniform([43, 40, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_55612ae909ad55ad56bb48a971449efc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01d55c39be47665b8027112537f006d1
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc219cf526e458de33df21ff9de38ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382b20b4b9f61800ddbe43e1cf3aa712
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_a6042736694a67eb8bc1528f5aa79ba9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d31eee549a312ba6de8ed90c60c01200
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e0242c1fa2029637fa51c2ec6c7c9569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7996d0f16b68a00175ac2ee58b32986a
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_7a46cbe338b4a1d0f038e7b5512567af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ccd931db1b27da5d28537e49231823f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_8213d8ea037e28908342d7077b7e8580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca583380b0e23044064e1e54c702ddfa
    def get_inputs(self):
        return [
            paddle.uniform([10, 60, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5878e6857a539b842619a8c91d77496f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 60800, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 60800, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_394aed8eb4c48dce8472bdea60696d48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5878e6857a539b842619a8c91d77496f
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 60800, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b7b153863ddbb310c1d51fdf89dd925a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 60800, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 60800, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0b42fc2e477d353649d220344ccf761f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7b153863ddbb310c1d51fdf89dd925a
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 60800, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b61243d5931a4a9212e7f907f98ad5c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 60800, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c1ecc82824204b100f2d59be2d4d14be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b61243d5931a4a9212e7f907f98ad5c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_97f46f2ddc769f1123ff6955cd790a0e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 49, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_78369d35a023e766a83c9ea6b0cd0056(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97f46f2ddc769f1123ff6955cd790a0e
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e7018c8290aea098c5a3f2ac31a3962e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 49, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f93f9e1ddf87342fd8d308ae94592522(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7018c8290aea098c5a3f2ac31a3962e
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_732a0463433346915e9e89daba81721a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_457c7db8651fc3c3a46615b2680f5cca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_732a0463433346915e9e89daba81721a
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8fc57001711b806a6526ca853ea32bd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05b9bb8eae4016c3fd8b4548ed85dab6
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_790b39e3e89f7e03563138cd27d97ce8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c45914fe8b40a284028e09cbdb673ee
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be67299aa43f002268392a6c49d469ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1677f2290efe9a742560ef7d7dfb3ba
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_76044ef22c2b100c88233705f0a0a9ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0399b6b5890bd2e28b6c4105967fdbad
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.49489855766296387], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3d28124ef9da8b74d34e51fa89c120f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d07132c8b6e809630ba97c8e007c0d37
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_57d13157208d60ca4bba4a608f1cd7e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 784, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 784, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_08518e09f71108972795f63c3845623d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57d13157208d60ca4bba4a608f1cd7e1
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_74df853cb3e234fdc02f2538a0f4a188(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 784, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 784, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_caca49919a414e392b247247391ffbd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74df853cb3e234fdc02f2538a0f4a188
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_566f215bc9b6fe9cbbcece1bb178a5b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 784, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ff172bb99fda2b29a788c66e30fd581e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_566f215bc9b6fe9cbbcece1bb178a5b8
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_94b0efc007e03ee2f1b76680461510ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_071ed06fac5df79e7278f63bd3047a99
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 21504, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_94b0efc007e03ee2f1b76680461510ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_071ed06fac5df79e7278f63bd3047a99
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 21504, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_33d29b4b280b656c19117380d97cf363(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea0be2fd07a5b31350bc3ce58eceaa31
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_51c40fa1586b97a36bcde0f702df97b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 49, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7d136372223a8a15f4c6a86a1d669690(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51c40fa1586b97a36bcde0f702df97b1
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7863b2beff958c4ac56a444880cee21e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 49, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_def03e62a4814d783b71ad29c5d28820(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7863b2beff958c4ac56a444880cee21e
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6703f810822211bca67b8219701506ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_331180ea98e0a80a458572ff2acd765a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6703f810822211bca67b8219701506ea
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a1d72b749c9e238b5f05268b163789e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_930e4f4171ba2d1a320df0a3a1340a56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1d72b749c9e238b5f05268b163789e1
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.5729997158050537]], [[0.5468035936355591]], [[-0.34526777267456055]], [[0.35399574041366577]], [[0.06473919749259949]], [[0.0907289981842041]], [[0.0012007057666778564]], [[0.36100754141807556]], [[0.0616908073425293]], [[-0.44871985912323]], [[-0.1183769702911377]], [[-0.06849408149719238]], [[0.4000604748725891]], [[-0.7309362888336182]], [[0.11584849655628204]], [[0.0638628900051117]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_cbf6a6b226fcf987b3727f42230e2416(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39670343df85f06e46d21b863131110d
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 80, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.6145999431610107]], [[0.6093606948852539]], [[0.4309464395046234]], [[0.5707991719245911]], [[0.5129478573799133]], [[0.5181457996368408]], [[0.5002401471138]], [[0.5722014904022217]], [[0.5123381614685059]], [[0.410256028175354]], [[0.4763246178627014]], [[0.4863011837005615]], [[0.5800120830535889]], [[0.3538127541542053]], [[0.5231696963310242]], [[0.5127725601196289]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class PrimitiveOp_f22bfff1740b78164f04182323f8f526(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 9216, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 9216, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ad803da8e079df212d5cdb63ff1b08a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f22bfff1740b78164f04182323f8f526
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cd82d6fe25ecea08a2e7b2453d9be0d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 9216, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 9216, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a64af391ce66845cd55fc54f5821dad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd82d6fe25ecea08a2e7b2453d9be0d8
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 9216, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4b34210e2314188fb68f9390d5303b4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 9216, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4efa539cf90d7a4a2f9a085288b5523b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b34210e2314188fb68f9390d5303b4c
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ec2442c3da67124857cfd77b695d1e8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_373f28edd813211622ed1dd23f15b0b4
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ec2442c3da67124857cfd77b695d1e8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_373f28edd813211622ed1dd23f15b0b4
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ec2442c3da67124857cfd77b695d1e8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_373f28edd813211622ed1dd23f15b0b4
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cebe559d7af21eb4b81f8dbd9110f87e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c514e2a5c05418dc0e1b53b10e40f6b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_86dadf4db7db692b6862fce544947266(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e08d7f1689a76970e8a0fa8557059a8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 21, 21], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_59a3185ecb207b8a35e34c0b35510afb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a6ac6e417aa2f9e70b7543858833ed7
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_d409b732256ad3b5609b1e4a089dbc50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8b788cb6625f4842bd8dcca7be8c085
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc219cf526e458de33df21ff9de38ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382b20b4b9f61800ddbe43e1cf3aa712
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_aff071a0ab5a91d43320ed090b201399(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad9b0a72a1a93921ab0c11808ad69640
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 46, 46], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_ccc23c738fe52a511e5d5b3f08368ff0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_321399b2c2bf46627156294e15157a86
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([551], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_3c0a86efd7e5554100115bb8fff63094(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be9379a2ffd678c991404990a3f835ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.44122177362442017], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d2c918f076d7d4f82ade29a3686b180d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cc472645e01aca6781df184b92f0cfd
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5e48406f87d8dfe173215c62b3b38d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a280b542eb2666a0f42ad377c7d05bb
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5a97e3c13d003fbdf95d3791703211c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8198545b41671385e47b0db8d12c7556
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_a6c012961493fa1de3a7e4276d26effb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092fc4de7b2e759352aca979a690cfef
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_c8cd99e444dad3dc78e88ca0a51adf00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab75d6f916b8c515969c8108422557d3
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d6172caada93e27ce0f496ad3beace0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_597a6361be11b23fe2d262c45726cc7a
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_27616106a6dd86d5554fe1a1875bd07d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c54270e10cb3f0d4753cec8cd8b9ecda
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_df3f0598f249e5ca08f920ce6d147745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dd8f5115e9b9b191e1a05315edd9c17
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_6116e1a6a2b5d12f0b1984e4ba532ceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fc13540db8c468ea0d8ee125772f421
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 256, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8fc57001711b806a6526ca853ea32bd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05b9bb8eae4016c3fd8b4548ed85dab6
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_790b39e3e89f7e03563138cd27d97ce8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c45914fe8b40a284028e09cbdb673ee
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be67299aa43f002268392a6c49d469ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1677f2290efe9a742560ef7d7dfb3ba
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d725ad75482a80a14e59b3b3c00fc205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f75c0dd4434232b4fd14b27654a58105
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_af4f9861c467d8f7839f7f035b2e81c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b6836b520a449ffacec29e71582bff5
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3df821dbf04205963b7a5e9accd1888f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2eb1af84e0b1a761cafa4073ef8f1cba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3df821dbf04205963b7a5e9accd1888f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_c95f33acdef3acb81a84e18cda872c6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b4cad3a8e65763df84c2b3237594a16
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 88, 88], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4fe93ab0db01dd2a2aaec1c7bbe3ddcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c19d801d8a9adac0c877ec38bfa79311
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_e2d66656b91e00ec133389b33b3f89e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e8077b2266ae65fb8ba255331ccd0a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_624b0608c3ee2834eae9d14d485dbfb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fcf03a6086130ae9486abb2f881899c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.19876134395599365], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_624b0608c3ee2834eae9d14d485dbfb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fcf03a6086130ae9486abb2f881899c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.19876134395599365], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_36bdedce90090f2b6bb4063f6dd4d7ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afd8fc5e9722ed38aa8e16a950fe160b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_f67304ad72b0f07677539fd291c507ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e955f135d79f9cab1a2d7bece733a436
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.4925087094306946], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_45b7123a3b8671aba01feb194d4d3630(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9755cc80340b3da84850fef2c74afd00
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_81d127e66eeaf3a06d2a06fdf9e5c504(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dd8f5115e9b9b191e1a05315edd9c17
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_80e2d970fce53c4289b48bdcfc42288a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f12c31ad5a0ed5599cd9083bee1205a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_afd4c331ad020e16f27dcd751fad3adc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e59d1345e52566a0de1041ad141a1c2
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_afd4c331ad020e16f27dcd751fad3adc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e59d1345e52566a0de1041ad141a1c2
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_afd4c331ad020e16f27dcd751fad3adc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e59d1345e52566a0de1041ad141a1c2
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_f7c2899ef3c153bd2a9abaeea4f7c685(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88524ba379499f55c3ea410fbee25873
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[-0.4899396598339081, -0.40941154956817627, -0.48135000467300415, 0.4581149220466614]]], dtype='float32').reshape([1, 1, 4]),
        ]


class TestPrimitiveOp_b31fe694663a63c77eaf9110c02771f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79fab2ddab9377dd34e70e31aa78037e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_e6c39c6655f342d022ad9d2d5fc95d70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d7803907307b1ef52ed725b0700a107
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_d10ea13b4d4128d62dfb4a76605a0060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b4af2e724b4b2010f490551ae44b0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d10ea13b4d4128d62dfb4a76605a0060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b4af2e724b4b2010f490551ae44b0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d10ea13b4d4128d62dfb4a76605a0060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b4af2e724b4b2010f490551ae44b0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d10ea13b4d4128d62dfb4a76605a0060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b4af2e724b4b2010f490551ae44b0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_523679b4ab520da2259255aa0c57dca2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ba5f2f9d7280059f0268c44cc5db73b
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8cd99e444dad3dc78e88ca0a51adf00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab75d6f916b8c515969c8108422557d3
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d6172caada93e27ce0f496ad3beace0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_597a6361be11b23fe2d262c45726cc7a
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_27616106a6dd86d5554fe1a1875bd07d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c54270e10cb3f0d4753cec8cd8b9ecda
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc219cf526e458de33df21ff9de38ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382b20b4b9f61800ddbe43e1cf3aa712
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_4fe0a513d8a446b2e80c5f4426a37902(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e14f6ebf089ccc1fb85f749a2c3c30a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_15bcbb6414ff84a0d699fdaafd45d2b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 160, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 160, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a7352c87413d91e5b8b998f2df3273ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15bcbb6414ff84a0d699fdaafd45d2b5
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_39231879b907c4e74184405b5f147722(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 160, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 160, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_da31bf5d6a04726a891465e7ec6ae534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39231879b907c4e74184405b5f147722
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 160, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9304bb250d6d76ef639efc1e1e0c1339(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 160, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_825bb9e1509de05282561681030b3295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9304bb250d6d76ef639efc1e1e0c1339
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b31fe694663a63c77eaf9110c02771f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79fab2ddab9377dd34e70e31aa78037e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_1c41f33b5fe464530fca6e055042dd6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5985f9b73ca924c927275de1d8fd671c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_b3e60cdfa232406185080e63b99aeb7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_376fde9d5c241f7d14927bb3fdda5fff
    def get_inputs(self):
        return [
            paddle.uniform([43, 112, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d45ba9b26809ee86adf3dadcc6e03895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7541fd6af8509f0ce8734aa60e436973
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_523679b4ab520da2259255aa0c57dca2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ba5f2f9d7280059f0268c44cc5db73b
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_a6eeee7e8399f3d6f87478e8bc1dce9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49cc3bcf5fe9bc7fdf0f0c1fbf48a019
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 16, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_aaf875efdac0fd8d18ab323ef076a730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([1.2373090982437134], dtype='float32').reshape([1]),
            paddle.to_tensor([-0.2829173505306244], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_60632bd6ac092d4a1b3a0a07d1d3d896(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7303234934806824], dtype='float32').reshape([1]),
            paddle.to_tensor([0.4355696439743042], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_e76985a4e90f64c2bd2027e7ac3541fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bf61eef8fe289dfd9d842a35208591
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e76985a4e90f64c2bd2027e7ac3541fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bf61eef8fe289dfd9d842a35208591
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e76985a4e90f64c2bd2027e7ac3541fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bf61eef8fe289dfd9d842a35208591
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e76985a4e90f64c2bd2027e7ac3541fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bf61eef8fe289dfd9d842a35208591
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_455217ed1d7009d313338b3eac3af37e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 169, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 169, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_69995ea9fafbb758741a1b1b65a73e56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_455217ed1d7009d313338b3eac3af37e
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_39b8a2f47e83ba679bf7b733e9fe326e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 169, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 169, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fd222b773ce8cc371b763c4e2eb953fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39b8a2f47e83ba679bf7b733e9fe326e
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 169, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_56ee0d01df62db762ac99ea5dc9d1313(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 169, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d664660aa4fd1ca78fa4a9ce259f09db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56ee0d01df62db762ac99ea5dc9d1313
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1024], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7c6dbc2a702ccee42af824d144d42c8a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 169, 2048], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_52959110d894e23f17cd47388ff3f4cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c6dbc2a702ccee42af824d144d42c8a
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.7071070075035095, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_882e6d37aec542a4e244f982ca1f0e3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c6dbc2a702ccee42af824d144d42c8a
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.5, dtype='float32').reshape([]),
        ]


class PrimitiveOp_fab07f3ae1e367e9d62d560a8b3d5d5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 169, 2048], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 169, 2048], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1b4d090606a4e227596a41428bbb8cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fab07f3ae1e367e9d62d560a8b3d5d5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 2048], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 169, 2048], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_69995ea9fafbb758741a1b1b65a73e56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_455217ed1d7009d313338b3eac3af37e
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fd222b773ce8cc371b763c4e2eb953fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39b8a2f47e83ba679bf7b733e9fe326e
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 169, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d664660aa4fd1ca78fa4a9ce259f09db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56ee0d01df62db762ac99ea5dc9d1313
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b0f82181b3adf2561c3103da1fb6335b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a11a838d9bfb17d4b04b2b87572d5082
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_2ae39a5b40bfcfa25cd64686edbb22f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da68c0f6e208c8f3ca8c5df3de35c420
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a5a49e9377e45b68a75720f0f70ebfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf6ae6ee44a92275d56a63ee17bb8b8e
    def get_inputs(self):
        return [
            paddle.to_tensor([2015], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e26d42fb0c7d435e5e79aef04ec4f437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e67d8c88c765684e9338b81f91013be6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e67d8c88c765684e9338b81f91013be6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e67d8c88c765684e9338b81f91013be6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e67d8c88c765684e9338b81f91013be6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_37ab475d04fe477a6d1411594711ffd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([2015, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_37ab475d04fe477a6d1411594711ffd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([2015, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e67d8c88c765684e9338b81f91013be6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2015, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_8b52c36df5d9cb80da71f086e5d714df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c53df8392ce8625ae243f2ee7b563a5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8b52c36df5d9cb80da71f086e5d714df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c53df8392ce8625ae243f2ee7b563a5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8b52c36df5d9cb80da71f086e5d714df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c53df8392ce8625ae243f2ee7b563a5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8b52c36df5d9cb80da71f086e5d714df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c53df8392ce8625ae243f2ee7b563a5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8b52c36df5d9cb80da71f086e5d714df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c53df8392ce8625ae243f2ee7b563a5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_3088a28bdf670a31e5677635c36f2c45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_583338ebcf3dbad8337070772726056a
    def get_inputs(self):
        return [
            paddle.uniform([1, 624, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 624, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f3d932ca3fab708661f9d8a4131a18f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1571de3b7504056e385f67f78d5a62f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_5b9e0e0986313e038f5e2c8823706ab1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c11bcb80120e9156b106a01d1a033d7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_18c2f187465c3e47e63c4530cd8aebe9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07c059a42edcf3de218e97da9fbac96f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.010511040687561035], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4a68ea25b83d6679299370ee7f9ec504(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 100, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e08a978daece187dd1935def49f8015(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a68ea25b83d6679299370ee7f9ec504
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_ae9bb4bb2845cdf088a7f4f093616f75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6a76a7eda1110b5ac220b554f1120a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 44, 44], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_b166243f313b4d3bdac27572964c7b86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1e5c99c02e707f89895497c2deb623a
    def get_inputs(self):
        return [
            paddle.uniform([11, 80, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_d45ba9b26809ee86adf3dadcc6e03895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7541fd6af8509f0ce8734aa60e436973
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_ed573526a5c07c30fc3d3e65ae777938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03175c544330f9d37e4e8afbbf3f1ca2
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 23, 23], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0d8d55e9b37336a6ae0258500ffa4c7b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 200, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_828d115ade718aa4fcde66272322f109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d8d55e9b37336a6ae0258500ffa4c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_ebf7b90650f0cbee51a7e7f90034b8df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22af0ad1c13eed434c8b50308b1d8241
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 44, 44], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f9003cabed674ae6f8dd36d665c9d0d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_211eae2939cc588637a4f35e8548c0fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9003cabed674ae6f8dd36d665c9d0d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_e98354dc54eb5ec46d01d5f57de4b8d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3b4cc4ea02253ae05aa28eba571068f
    def get_inputs(self):
        return [
            paddle.uniform([171, 60, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 60, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_46f0e721caa6ba927b4d65e64edf8ad6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3177f9dd01ebc8e21c5a191dbebd0630(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46f0e721caa6ba927b4d65e64edf8ad6
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3177f9dd01ebc8e21c5a191dbebd0630(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46f0e721caa6ba927b4d65e64edf8ad6
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2eb1af84e0b1a761cafa4073ef8f1cba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3df821dbf04205963b7a5e9accd1888f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_28c0b60fe061afedc9758ba7b49cbdbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc3ec5b0206724b7e1f435e15a92fd19
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_b2981cc22235bf96425b52d79f0ed165(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7362f7afa42952ba39dab202622b7c5e
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_1138f2f484ffda0d688181e6ab0d12e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3202affc3aba5367ae9a98124a1547b7
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_61b2b2cafb18a008946f81fd5dc66467(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32768, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 32768, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ec75498a99a6abbcf364ce06532e0437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61b2b2cafb18a008946f81fd5dc66467
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 32768, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4c538be846f902febebbc7053f2ced04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32768, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 32768, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_670dcf22ec8acef0bf26f123494fbfaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c538be846f902febebbc7053f2ced04
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 32768, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5f7eced54ec138e4e0398b901826020b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32768, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e693b8b19f2cd84fda4273712aeee2c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f7eced54ec138e4e0398b901826020b
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_19d6887c242d5b454eb0fc93d223a17a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 320, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 320, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f299bf8db4d3ef58ab6b97fe27a4d977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d6887c242d5b454eb0fc93d223a17a
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b69696aecf71a1144fbc6bd6e027ab86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 320, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 320, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8a0d50245c6c7f2c4151dd2899334de9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b69696aecf71a1144fbc6bd6e027ab86
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_cff4421ec94fec32773214adaf02baab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 320, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cf5545733de5d34e5b11306386cdc997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cff4421ec94fec32773214adaf02baab
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_c8568c12e9f1a444d0914f64386f0190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c69efc00629a6fa23d714fdb6e006e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8568c12e9f1a444d0914f64386f0190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c69efc00629a6fa23d714fdb6e006e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8568c12e9f1a444d0914f64386f0190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c69efc00629a6fa23d714fdb6e006e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8568c12e9f1a444d0914f64386f0190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c69efc00629a6fa23d714fdb6e006e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_990786bb38024e86bf09b9526a4019cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23efb7705a00d013dde414a35a50893b
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_eb7d6e06a9517b9196329d5bfe03d7c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bdb49101bf446c654cc08ec506e2a3e
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_ab7bcdb80bc608eff933ff9d091da22c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3a875073a11ba5ab64d9b4598a017dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 38, 58], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.23888331651687622], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_b2d4d7a3a22170d040d1a37b98cbb112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a64fa566b317e48c6abbc2182f3c730e
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_18c3ef78bd190a43e20df7bf3b924854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6091199a470ab6f8f7a4e1eee4c94ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b31fe694663a63c77eaf9110c02771f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79fab2ddab9377dd34e70e31aa78037e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_a8c3f0b7551f7f65115c010a09eb3f78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f833bb4f34910f3c00dfabd43cacf512
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 96, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_112245adb0d2b325192705e26c77c490(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a100a88fd07868b4e4af1eb096d6600f
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_112245adb0d2b325192705e26c77c490(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a100a88fd07868b4e4af1eb096d6600f
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_112245adb0d2b325192705e26c77c490(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a100a88fd07868b4e4af1eb096d6600f
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d409b732256ad3b5609b1e4a089dbc50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8b788cb6625f4842bd8dcca7be8c085
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b4db3448ad49d516bd7300726b009e7e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 256, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e80ba68e409a4dad587398f97e31a34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4db3448ad49d516bd7300726b009e7e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_1f77178fcfcb5907dac6ce17c63cdab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6f4f2695ad16c7c77e69861404b436
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 1, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_e029cdeb3f10016aca87f8e4f50a4958(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f36b1c24ce4555f892835091fa7d24d
    def get_inputs(self):
        return [
            paddle.uniform([145, 36, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 36, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_5846997ad303bb3f1611c37032658b33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8253ad6111de39b8930be7b782affcb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5846997ad303bb3f1611c37032658b33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8253ad6111de39b8930be7b782affcb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5846997ad303bb3f1611c37032658b33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8253ad6111de39b8930be7b782affcb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5846997ad303bb3f1611c37032658b33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8253ad6111de39b8930be7b782affcb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5846997ad303bb3f1611c37032658b33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8253ad6111de39b8930be7b782affcb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_956b6307dd67e01353d2fc0dde916a64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe14fb69ea018dddceaaba5c0c8d4222
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b31fe694663a63c77eaf9110c02771f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79fab2ddab9377dd34e70e31aa78037e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_2d1d5f62da7c7f4489e3a9b9a0621ac8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ec5685eb7ecd374ae0a21bd83d0d5c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_93f9e380471d9da5edbfc89bfe864c45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b860cca7d74e41392aecd7a90ee59238
    def get_inputs(self):
        return [
            paddle.to_tensor([2.3922815322875977, 2.074402332305908, 2.0009725093841553, 1.82877779006958, 2.5674736499786377, 2.1610472202301025, 2.320549726486206, 1.6990677118301392, 1.8623496294021606, 2.0880379676818848, 2.2874526977539062, 1.7015490531921387, 1.9379594326019287, 2.656717300415039, 2.359018087387085, 1.6493213176727295, 1.944158911705017, 1.9806581735610962, 2.001063585281372, 2.5149178504943848], dtype='float32').reshape([20]),
            paddle.to_tensor([0.9431471824645996, 1.3167967796325684, 0.9454124569892883, 1.4895567893981934, 1.04916512966156, 0.6405051350593567, 0.8531206846237183, 1.3808798789978027, 0.5210181474685669, 1.4319356679916382, 0.5749496221542358, 1.130189299583435, 1.272364854812622, 0.7284885048866272, 0.7370707988739014, 1.1263890266418457, 1.374311923980713, 1.2648186683654785, 0.7743330001831055, 0.9311091899871826], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_8be6005065efdee3c04dffb20f073c3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b860cca7d74e41392aecd7a90ee59238
    def get_inputs(self):
        return [
            paddle.to_tensor([2.119749069213867, 1.767011284828186, 1.7605992555618286, 2.029568672180176, 1.7254993915557861, 2.394219398498535, 1.7988717555999756, 2.1211276054382324, 2.2528650760650635, 2.3174240589141846, 2.512205123901367, 1.938329815864563, 2.3324379920959473, 1.8674839735031128, 2.5236334800720215, 2.599437713623047, 2.705566644668579, 2.409569025039673, 1.7132900953292847, 2.3739190101623535], dtype='float32').reshape([20]),
            paddle.to_tensor([0.05685281753540039, -0.3167967200279236, 0.05458754301071167, -0.48955684900283813, -0.04916509985923767, 0.3594948649406433, 0.14687931537628174, -0.3808799386024475, 0.4789818525314331, -0.4319356679916382, 0.42505037784576416, -0.13018929958343506, -0.27236491441726685, 0.2715114951133728, 0.26292920112609863, -0.1263890266418457, -0.37431198358535767, -0.2648187279701233, 0.22566699981689453, 0.06889081001281738], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_d5f889ce84dcfcfd9bfeb39ce556cb67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b860cca7d74e41392aecd7a90ee59238
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5941967964172363, 0.5429457426071167, 0.4969627857208252, 0.4326198101043701, 0.6522173881530762, 0.5612178444862366, 0.5609815120697021, 0.3845783472061157, 0.5123498439788818, 0.49723950028419495, 0.5957459211349487, 0.41768068075180054, 0.4576292932033539, 0.6106078624725342, 0.6005750894546509, 0.3823092579841614, 0.4147886633872986, 0.46676862239837646, 0.48403066396713257, 0.6263010501861572], dtype='float32').reshape([20]),
            paddle.to_tensor([-0.12868961691856384, -0.353333979845047, 0.051278889179229736, -0.4334791898727417, -0.4992327094078064, 0.10352194309234619, -0.03556165099143982, -0.21199136972427368, -0.34256961941719055, -0.04708665609359741, 0.29697930812835693, -0.12769043445587158, 0.14123564958572388, 0.24236905574798584, 0.23481661081314087, 0.154252290725708, -0.33004313707351685, -0.43981418013572693, -0.15075603127479553, -0.3905327022075653], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_6af6cf0e4f054c03852c15733fa911ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73ac9e1788bcb88b4f8ecb5ae2b2ae40
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_3657b4ce2eea777f48ef6575611ec396(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c63c960682dda3a4cada933be990e9d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.4990195035934448], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_d6a320239f0784728ad1b49ca4ceb367(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71914385028c60457327f20b50af0ff7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.5591869354248047]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.13120320439338684], [0.1371973752975464], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_3ce8ffcd50ee9c80a645d5ee50460063(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71914385028c60457327f20b50af0ff7
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.38926953077316284], [0.4182015359401703], [0.14763367176055908], [-0.011704206466674805], [0.5591869354248047]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.971923828125], [0.4351953864097595], [-0.2429632544517517], [-0.15176233649253845], [0.06258434057235718]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_a5e0777a73bcb5e1c4bc8d3b82c88e14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71914385028c60457327f20b50af0ff7
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.6303538680076599], [-0.7695514559745789], [0.23850399255752563], [-0.3411584794521332], [0.7801095247268677]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.13120320439338684], [0.1371973752975464], [-0.04692918062210083], [0.02536982297897339], [-0.47886979579925537]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_efdf33bdbecf8d0d9f66fb14a5acd255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71914385028c60457327f20b50af0ff7
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.38926953077316284], [0.4182015359401703], [0.6210484504699707], [0.3223789930343628], [0.7801095247268677]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.971923828125], [0.4351953864097595], [0.06543827056884766], [0.027780622243881226], [0.06258434057235718]], dtype='float32').reshape([5, 1]),
        ]


class PrimitiveOp_a29acfc06c8f4e491172ee7d6e889022(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 65536, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 65536, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_18d8cb7593d387a904e66ecb6967cd8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a29acfc06c8f4e491172ee7d6e889022
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 65536, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f2a52ca9ae0d40989f0b067ee4545601(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 65536, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 65536, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_626ce0163c43455b48b640525c44720e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f2a52ca9ae0d40989f0b067ee4545601
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 65536, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e1c97deec9cc31effb8798ed0061821b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 65536, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8452a49e2460128ca44dc9374a1f0018(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1c97deec9cc31effb8798ed0061821b
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8407e2d8d2f2d788039899ada63862c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba5cc92704ec940eeadc35b0cfa3e2ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_6cbe2c84d87796bced1f4fd511e4c35a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f16b2d4d9a319dfa9e4a7b0b6bcfee2
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_4429320551cb0942112909f9ea1f703a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c135422a2a2ad85191f982895874914
    def get_inputs(self):
        return [
            paddle.uniform([15200, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15200, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4429320551cb0942112909f9ea1f703a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c135422a2a2ad85191f982895874914
    def get_inputs(self):
        return [
            paddle.uniform([15200, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15200, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ec0d0c7bdad034abdadb4a00c6897706(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[15200, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1bf18fbb12b834a4a8cec7b32b6bf274(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec0d0c7bdad034abdadb4a00c6897706
    def get_inputs(self):
        return [
            paddle.to_tensor(-1.0, dtype='float32').reshape([]),
            paddle.uniform([15200, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4429320551cb0942112909f9ea1f703a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c135422a2a2ad85191f982895874914
    def get_inputs(self):
        return [
            paddle.uniform([15200, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15200, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2967427d77a41a1132cf46c7eaf1bec8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf6ae6ee44a92275d56a63ee17bb8b8e
    def get_inputs(self):
        return [
            paddle.to_tensor([1830], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_77dc0f987fce49d3de759baa7291f747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eb65a6e9b503b24ded4af763cb307408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eb65a6e9b503b24ded4af763cb307408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eb65a6e9b503b24ded4af763cb307408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eb65a6e9b503b24ded4af763cb307408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3e71e7c0fec4f7f665ab88e5a70a0e02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1830, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3e71e7c0fec4f7f665ab88e5a70a0e02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1830, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eb65a6e9b503b24ded4af763cb307408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1830, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e0242c1fa2029637fa51c2ec6c7c9569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7996d0f16b68a00175ac2ee58b32986a
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_9287b1dac0bfe69bdd6fdd3cdc742f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a5eb7010acdccc0e8dd26e43e3813f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 34, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_40fbd6080b85fe32b9d63842ca51d8a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_391ea07d377bd44a525315beafa2d798
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 1, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_0cea6572670fc2f649017c3999918d29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22a4d1373b2bac1f63849563cdaff296
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b2b72ac8d18845b6918a4f73ff406e5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9454b44bdcd589682ee445171b8ec762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2b72ac8d18845b6918a4f73ff406e5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_887b5d76dce70cbf4012ee8e36183af6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7111f13fb5b41f2eca68466d2f7310d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_887b5d76dce70cbf4012ee8e36183af6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_166b5181f9a91c9f15d46e44f9d2bcc3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8817b65d9695e7a0c2d68315a09963f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_166b5181f9a91c9f15d46e44f9d2bcc3
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_1aa15be6724fe96140d9ee1a2578eb00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bf619cdf78e852feb559b5e72a5ba77
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.0802028477191925], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_9c8b7837105ea4b71f18120cb273bb6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06afa0ac8681719902f3e86d0feb1b9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.32802125811576843]], [[0.49818792939186096]], [[0.5841472148895264]], [[0.3071761131286621]], [[0.5178560614585876]], [[0.5419326424598694]], [[0.4845777153968811]], [[0.4558437466621399]], [[0.3680122196674347]], [[0.5982741117477417]], [[0.6109840273857117]], [[0.4478115737438202]], [[0.518684446811676]], [[0.5572967529296875]], [[0.47246113419532776]], [[0.49507418274879456]], [[0.6755238771438599]], [[0.5881323218345642]], [[0.45370736718177795]], [[0.40743738412857056]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


class TestPrimitiveOp_e8d878ebb439c20b2ae99967473a9d37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08a35fac70f021394b01b6711933f04f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_13bf0e2e5799f4d87f6e27cb6ed912cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34967eb0bb98058b9f8a9a8cc0d2f7e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_ba1d74b3909f7642347bf132cca90db7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7852bc1b6dcc7869f2cc107389bc5f0b
    def get_inputs(self):
        return [
            paddle.to_tensor([2.1307504177093506, 1.836279273033142, 2.3008804321289062, 2.6398448944091797, 2.488538980484009, 1.899686574935913, 1.7902967929840088, 2.67394757270813, 1.6935261487960815, 2.355630874633789, 1.7174203395843506, 2.107821464538574, 2.568502426147461, 2.1975748538970947, 2.2575485706329346, 1.9274046421051025], dtype='float32').reshape([16]),
            paddle.to_tensor([0.5223820805549622, 1.1309725046157837, 0.8182364702224731, 1.388822078704834, 0.9583341479301453, 0.6766443848609924, 0.6133625507354736, 1.4313998222351074, 1.198455572128296, 1.3228504657745361, 1.1049326658248901, 1.26625394821167, 1.3995301723480225, 1.0191138982772827, 0.6602637767791748, 0.8872633576393127], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_1c55b092939389e544a623c653f160a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7852bc1b6dcc7869f2cc107389bc5f0b
    def get_inputs(self):
        return [
            paddle.to_tensor([1.8647176027297974, 2.3728384971618652, 1.8160676956176758, 1.8408079147338867, 2.287106513977051, 2.0656750202178955, 2.3176286220550537, 2.668424129486084, 2.2012877464294434, 2.5427544116973877, 2.1444454193115234, 2.173513889312744, 1.870514154434204, 2.0755252838134766, 1.7141977548599243, 2.2043871879577637], dtype='float32').reshape([16]),
            paddle.to_tensor([0.47761791944503784, -0.13097253441810608, 0.18176352977752686, -0.38882213830947876, 0.041665852069854736, 0.32335561513900757, 0.38663744926452637, -0.4313998818397522, -0.1984555423259735, -0.3228505253791809, -0.10493266582489014, -0.2662540078163147, -0.3995301127433777, -0.019113868474960327, 0.3397362232208252, 0.11273664236068726], dtype='float32').reshape([16]),
        ]


class TestPrimitiveOp_de0c79ac8f762a889d87bf66d67fe702(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7852bc1b6dcc7869f2cc107389bc5f0b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5009220838546753, 0.4415011405944824, 0.5531898140907288, 0.7376319766044617, 0.6200365424156189, 0.48833996057510376, 0.49854576587677, 0.6690825819969177, 0.3981895446777344, 0.573804497718811, 0.4181528687477112, 0.5225826501846313, 0.7118425369262695, 0.5499768853187561, 0.5182381868362427, 0.4896577000617981], dtype='float32').reshape([16]),
            paddle.to_tensor([0.08473503589630127, -0.48493167757987976, -0.17284682393074036, 0.11406326293945312, -0.3063887655735016, 0.12536591291427612, 0.23882931470870972, -0.3631182909011841, 0.04979729652404785, -0.07877427339553833, -0.35734468698501587, -0.14594513177871704, 0.459084153175354, 0.4370843172073364, 0.20332396030426025, 0.25458383560180664], dtype='float32').reshape([16]),
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


class TestPrimitiveOp_694038187e6b55faa0232a237d63f092(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([4850], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4850], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_694038187e6b55faa0232a237d63f092(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([4850], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4850], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cdb6604d56aa3690449a420f12aa9962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1b34b55aabf7993d26444c37fd95b6f
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_eea475d3d3732da9d900e5d828e40b71(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 2304, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 2304, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c8c92bcb0b955f1664b46bc81b3c1830(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eea475d3d3732da9d900e5d828e40b71
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 2304, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4ecd2a7c83fb4c7fc602a0b6759c179d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 2304, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 2304, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4182b3d7937ee177a4f155f78d637279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ecd2a7c83fb4c7fc602a0b6759c179d
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 2304, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_41acc07e1c247a5a14abef3d4dde22ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 2304, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_921c8b03e50940004b512d41faff49d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41acc07e1c247a5a14abef3d4dde22ee
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f3d932ca3fab708661f9d8a4131a18f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1571de3b7504056e385f67f78d5a62f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_e29a633385e84cbf25baf3b52d9f497d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e5b08f724933040792e48be5419527f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_ffce074b2347e1c0584570a7c2a05092(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3efe362643d10e20d0d525ff9c7618e
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_81d127e66eeaf3a06d2a06fdf9e5c504(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dd8f5115e9b9b191e1a05315edd9c17
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_87a9c0d4cb8a5822a572b43846980d33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8537dfae7567de2fa5575c59c20debe
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 68, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


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


class TestPrimitiveOp_d38bdb5e14c21f8bb9d48e9cc873a71e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea3ad6a4af220e8de872eaa59cebe711
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f7022f436bb7c9ed007acdaeef97b3e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21760, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 21760, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1cc9b75c1767839dd80be3578eee8bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7022f436bb7c9ed007acdaeef97b3e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 21760, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_745494a1f325b6fa3c8acfdb4f521c3b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21760, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 21760, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_78906be92c6d42c8ef4ea03c8069c3ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_745494a1f325b6fa3c8acfdb4f521c3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 21760, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_657dfc96ffbb62f3c63b84c2293c94ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 21760, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_be62255c17dc0c0fae04cac0e9010876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_657dfc96ffbb62f3c63b84c2293c94ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_6436ec79f69419d92f8b4e8b4dc7db5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa34e62f4f2ceafc1c2d9505b9015908
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 1, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b31fe694663a63c77eaf9110c02771f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79fab2ddab9377dd34e70e31aa78037e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_eeb90f810a081d36e5971340f1a8366e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67c444cd82fc1906f1b4ead121f105db
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 256, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_458a762b3756eb03818a0411ab130382(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d431583dbce2806d5aea6fac77553e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.2513127326965332], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_b54fa9926393e0acd0654965f97e03e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad4d8b2af8f1e48e6c666ef31b8c2059
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 1, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b8466eb47b52685dbfba8b3db225aeb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 198, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[54, 198, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5b57a6b8f7e6ba46ac1483fdaec08daa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8466eb47b52685dbfba8b3db225aeb7
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 198, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7af37a767c16889b3ccad6768087a621(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 198, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[54, 198, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9096cd9f92012b8a083ade6a273e7a6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7af37a767c16889b3ccad6768087a621
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([54, 198, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6135c961307ff0945025fe4949272fcd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[54, 198, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6bf4889b6b0b01ccb67a6939a49d4832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6135c961307ff0945025fe4949272fcd
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc219cf526e458de33df21ff9de38ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382b20b4b9f61800ddbe43e1cf3aa712
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_01077e5c6483a7941d104d50c9919f8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ada4132836aa661971ac9d154b79896
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 19, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f719f52567c6f8da3c9868fec16b631c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf6ae6ee44a92275d56a63ee17bb8b8e
    def get_inputs(self):
        return [
            paddle.to_tensor([3039], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_25e92270df71122670435240294f3dbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_09d6e0f8892638419eda95c2ee0cac80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_09d6e0f8892638419eda95c2ee0cac80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_09d6e0f8892638419eda95c2ee0cac80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_09d6e0f8892638419eda95c2ee0cac80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7c060395c04f8930c5768e5ad5a4edad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([3039, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7c060395c04f8930c5768e5ad5a4edad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([3039, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_09d6e0f8892638419eda95c2ee0cac80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3039, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4fe93ab0db01dd2a2aaec1c7bbe3ddcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c19d801d8a9adac0c877ec38bfa79311
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_cd9fdc6138d4d9f8358cb658b65e3070(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67448f2337e9c3d3bcfd73b04524295f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_7db91a8ff2795a6877e737980c6964af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626ae5005bb74d0d19e833cd0b4aec1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7db91a8ff2795a6877e737980c6964af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626ae5005bb74d0d19e833cd0b4aec1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7db91a8ff2795a6877e737980c6964af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626ae5005bb74d0d19e833cd0b4aec1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7db91a8ff2795a6877e737980c6964af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626ae5005bb74d0d19e833cd0b4aec1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_aaab0cede14ba50e5db05051a2590c0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f458c247bbde0b501f5ffb146114f4ae
    def get_inputs(self):
        return [
            paddle.uniform([171, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e85410b3ed2a5e86a932c73669c9aa5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1024, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_104c6752ab2e71a8a7221a0e739715ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e85410b3ed2a5e86a932c73669c9aa5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1024, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f1142d37cc1a1b80002123cbbbb12274(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1024, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ae1225412ad3dc6ca5be031a091d4a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1142d37cc1a1b80002123cbbbb12274
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1024, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1fe536fcdcf6942df3bf45e56ccfb478(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e14c5796a6b01106c869f76ddfce171e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fe536fcdcf6942df3bf45e56ccfb478
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_9858fe36835d0cefdfd759d65464119c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91dc404a62a9ef53dc011df69dc0346f
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_08518e09f71108972795f63c3845623d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57d13157208d60ca4bba4a608f1cd7e1
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_caca49919a414e392b247247391ffbd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74df853cb3e234fdc02f2538a0f4a188
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ff172bb99fda2b29a788c66e30fd581e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_566f215bc9b6fe9cbbcece1bb178a5b8
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e0242c1fa2029637fa51c2ec6c7c9569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7996d0f16b68a00175ac2ee58b32986a
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_60b620c2f1990467109fd92a0e6e7655(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2df90e7fe7ffdc7f4eb66c6e3d30b961
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_4739daef4f37ffb70bf56efefefad185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f904b4249c94af8f5bd043b44f301a7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.28747814893722534], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_4739daef4f37ffb70bf56efefefad185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f904b4249c94af8f5bd043b44f301a7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.28747814893722534], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_83993868cd77da7abb1b55265d4cece0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_260c21a37e265b18f9213c152646cf98
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_032d79ec70a94eaac23c32068abf4f18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a20ef40aebeb098fea9c3756d8eee753
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.3002668619155884], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_7dec6207574e24156d11669f3ed704e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9e31009351162921b8651dea231e5cd
    def get_inputs(self):
        return [
            paddle.uniform([10, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_61974deeb2d3a4aa863bbf228e52e4dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_91c81fe6df3862f592f7a40d56451293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61974deeb2d3a4aa863bbf228e52e4dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_19087f91a33c98d1b1df64595a808a03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f1f2ce33f99bc14e6cbe4b9b49b6cf3
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_41ec89f6832fbc97bd427594cbcd5775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5af49e0f99aa9f32c14a8800eafca8d
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c104dd895cba4f3f20bf661588c3424d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e948c5ec06727a5553c93d6b9d5e514
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_812ad552a1a75b71c984cc9c81c4aeff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f99381c513c67f933e83be887b62389
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_63360cafb2212617ba7d2e44ffc90494(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f98ea5d32539ef9702904a3797f06e1e
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_63360cafb2212617ba7d2e44ffc90494(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f98ea5d32539ef9702904a3797f06e1e
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_63360cafb2212617ba7d2e44ffc90494(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f98ea5d32539ef9702904a3797f06e1e
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_2fbc0ef8549bee4229328024159d5b05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f4316e09ac823e38388127c20d36206
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.07460933923721313], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6436ec79f69419d92f8b4e8b4dc7db5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa34e62f4f2ceafc1c2d9505b9015908
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 1, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b31fe694663a63c77eaf9110c02771f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79fab2ddab9377dd34e70e31aa78037e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_3a61d15f37e0025438f92031b5edb54d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3683b728cb38e102bc836e28eeeea7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 68, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_64dd70942b2bc701cfbe4938d1fe843d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3f803a756869e501116ea42698960ea
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1855833d1a5d0b933bfb611165712f81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_057715d92dae638dcf6709c1155905fc
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_57bee0f8676a7a7f95e587c0f67f4309(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8bcd5775fc0db4d50eb321309514e9e
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_e493f7e21d45a90ffd91bd7f82d746b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90412ca909e2194393aa35bb44fdba24
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.471342533826828], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e493f7e21d45a90ffd91bd7f82d746b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90412ca909e2194393aa35bb44fdba24
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.471342533826828], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_4400f0b817d87b69f5212689f6825a7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aefdace9d19840fd2c02dda73359eba4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_5bb7e4aeb2c2f6541893e9fa909affbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d7f35f039e7ad2049eb45bad5e4b525
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.4338390827178955], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_80af52dfe292d72107a04668b0cd1343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5328717a8cee22cfcdd245b743fb930d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_ab124ae4558dbde6206e6e4a94b136e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ec6cd942b64db2c8fcf7d19a07cf177
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_548f1bfd237c2afd376a088a6978a4d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61e2a73b85b68cf70c72f6ba6e376320
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.2378315031528473], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.1000869870185852], [0.10020291805267334], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_4a463ca20b3a66752d48b57fb44fae1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61e2a73b85b68cf70c72f6ba6e376320
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.17345154285430908], [-0.006497859954833984], [-0.07110410928726196], [-0.4251040518283844], [-0.8504189848899841], [-0.21283549070358276], [0.5175081491470337], [-0.539016842842102], [-0.7484806180000305]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.44422104954719543], [0.7355712056159973], [-0.05939823389053345], [-0.30565184354782104], [0.5283401012420654], [-0.5377517938613892], [-0.6035856008529663], [0.08256009221076965], [-0.3163154125213623]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_0884811824ddc86f0193b41d346d491e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61e2a73b85b68cf70c72f6ba6e376320
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.00872790813446045], [-0.16432955861091614], [-0.5041313767433167], [0.2942051887512207], [-0.07921814918518066], [0.03406834602355957], [0.3460357189178467], [-0.07707390189170837], [0.03394120931625366]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.1000869870185852], [0.14259999990463257], [-0.2612619996070862], [0.11226284503936768], [-0.31859248876571655], [0.024875521659851074], [-0.23641228675842285], [-0.3579455018043518], [0.2604370415210724]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_4f5f2db59a187a9ebe62998a67741d34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61e2a73b85b68cf70c72f6ba6e376320
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.00872790813446045], [0.30275672674179077], [-0.02017509937286377], [0.2942051887512207], [-0.028181731700897217], [0.35811108350753784], [0.6257123947143555], [0.15678679943084717], [0.050099968910217285]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.44422104954719543], [0.7779682874679565], [0.13703501224517822], [0.11226284503936768], [0.5283401012420654], [0.35644954442977905], [-0.01455610990524292], [0.2059648036956787], [0.2604370415210724]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_9e08a978daece187dd1935def49f8015(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a68ea25b83d6679299370ee7f9ec504
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_9e023160bfff9bd2d90dc758ac00c6b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbb9b388a9ca6cc168775c14411f09d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 18, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2ad803da8e079df212d5cdb63ff1b08a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f22bfff1740b78164f04182323f8f526
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a64af391ce66845cd55fc54f5821dad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd82d6fe25ecea08a2e7b2453d9be0d8
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 9216, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4efa539cf90d7a4a2f9a085288b5523b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b34210e2314188fb68f9390d5303b4c
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_91c81fe6df3862f592f7a40d56451293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61974deeb2d3a4aa863bbf228e52e4dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_b56f8efc8a207812edb49701ad7231e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d1ee8bbdd03eebeb42367a09f3b6c7c
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_0c26b50b24dfe91560f351ce78bf46f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5aad34d7c86157d1487993f25703db5
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.14821290969848633], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0c26b50b24dfe91560f351ce78bf46f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5aad34d7c86157d1487993f25703db5
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.14821290969848633], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_25e76fc7f2b2ad571e4c2f6904d5070c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8573bc190e2cdda526bce2f19a07883c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_84211788b980479448237a37ab558b56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e2e2d1970fd88b18fd9d1b4d7b374ba
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.33030200004577637], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_4681047ff92aef0deec628951357c2a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 49, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_86bae15ea8e123e080709e45e5a972b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4681047ff92aef0deec628951357c2a2
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_60a6a9d5d73b03004e18a33ade3fb789(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[43, 49, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9fd630d22fa1284e6cf55fa9657803c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60a6a9d5d73b03004e18a33ade3fb789
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e47583a69e57b4d722fda1acf601cd94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[43, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fd13547b0e011482049eabd6c38ee327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e47583a69e57b4d722fda1acf601cd94
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee0952ffa4beabf359a88c9149d51914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_458ab14b7d947a9d88ede4190462b605
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc219cf526e458de33df21ff9de38ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382b20b4b9f61800ddbe43e1cf3aa712
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_39e5f279924b73b81d4b35e97aeed661(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82aedc620f8680bcf4a39a1c8efd8eb0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_36363d7e097a47568edd8771d4a9d17a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2481e105c4881796aef69e0dd25a9d25
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_f9f27202cffaf32f3c494bad206e13ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73356518337e2f5c65479b43248a9b8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_18c8ea28ed5f019dabcf835b29a04d02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2f5611870d3f1d8af810feb55e461d
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_80af52dfe292d72107a04668b0cd1343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5328717a8cee22cfcdd245b743fb930d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_84efa98fa996c84a63d47fb98d7d85af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_538500537929b6b1c466453ae69ded23
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 76, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_60a34fa131d38bd74b1985da6eb9b8e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64b3661e08a9412b02da4daf9ca243b1
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 1, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_60b620c2f1990467109fd92a0e6e7655(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2df90e7fe7ffdc7f4eb66c6e3d30b961
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f8a7b82d462f2107ba9f271eabb11776(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 50, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 50, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aaac6ea7ca2fbda99923540dcb8beb16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8a7b82d462f2107ba9f271eabb11776
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_dcedb87a3d86cde827f037ac3eea584e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 50, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[10, 50, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_09ea9eff0dbc5c5118e099bec359063f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcedb87a3d86cde827f037ac3eea584e
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 50, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a3b25d945a95df602c53bcc90fe0b3be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 50, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6e25053dd9a75fe59c592df9e1a00251(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3b25d945a95df602c53bcc90fe0b3be
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f3d932ca3fab708661f9d8a4131a18f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1571de3b7504056e385f67f78d5a62f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_0a9436b2bde26c18dab1b4c97f7f5007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_535ac6c81c54fa149b44dd4f08f0cbf8
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 11, 11], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d725ad75482a80a14e59b3b3c00fc205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f75c0dd4434232b4fd14b27654a58105
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_4e98bfaf495055d999cff3b5fb3055de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ea64f73170f3a50b33c2c158d5e402d
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 34, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_990786bb38024e86bf09b9526a4019cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23efb7705a00d013dde414a35a50893b
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_a93552ff5a6a338faf2aa81ba05f28d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_548273f689e30af802afc00c29cff5d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_065fb0f371a07b8a450514681de79049(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53d856c85717ca4bec7292dee1ab55f9
    def get_inputs(self):
        return [
            paddle.uniform([11, 40, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_2eb1af84e0b1a761cafa4073ef8f1cba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3df821dbf04205963b7a5e9accd1888f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_f26fe24b27696535594497d51ceebff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c2d0ef34d18b466423e4abfe815aed4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_123d129ef1079efd3b36a7c6710f06e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39c493482373db37b12ad1412d08f9d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123d129ef1079efd3b36a7c6710f06e0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eab687913e0e3d772d31dbf05424e461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0f893a89be09dd2f9000d61202b6d7f
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e19dee740eca47ae59647d3ffe4a8d0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12df68d89b8813493227a5d5192e489b
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10e194c8bac5598853dea89b168480b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05eb97b286a792a80f7726c6363886ec
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_535d390e7ded077df6fc3fe0e3e15ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd27cb57db362f26d74af4bb48557b9b
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.0016358494758605957], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_f0c5565ea53aa0bc8d459c37819962ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f80d0ab208ce3eed5377f86b6cbe16fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a79afd61a8a00e72608f69b365fb808f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf6ae6ee44a92275d56a63ee17bb8b8e
    def get_inputs(self):
        return [
            paddle.to_tensor([2046], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_f5f69c197a5631b031de91f6232f4a01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_90aa029cecd67147e94b4ba267c06087(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_90aa029cecd67147e94b4ba267c06087(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_90aa029cecd67147e94b4ba267c06087(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_90aa029cecd67147e94b4ba267c06087(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3fe90bca81b927996c415728144b173e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([2046, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3fe90bca81b927996c415728144b173e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([2046, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_90aa029cecd67147e94b4ba267c06087(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2046, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_f4b0a1fad3adad2d81b2bb48c225ac15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a925e822f3f30e75c3ce588dc94246f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.15609928965568542], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_828d115ade718aa4fcde66272322f109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d8d55e9b37336a6ae0258500ffa4c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_a9b4085c921a9945456d5d82246fcce4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fcdb0534aaf89be4838b4682250534cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2eb1af84e0b1a761cafa4073ef8f1cba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3df821dbf04205963b7a5e9accd1888f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_44cb938c12df4a1564b35b7beb0094db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa3b319a493f507b319a5ef0d7b4cc68
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8568c12e9f1a444d0914f64386f0190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c69efc00629a6fa23d714fdb6e006e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8568c12e9f1a444d0914f64386f0190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c69efc00629a6fa23d714fdb6e006e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8568c12e9f1a444d0914f64386f0190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c69efc00629a6fa23d714fdb6e006e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8568c12e9f1a444d0914f64386f0190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c69efc00629a6fa23d714fdb6e006e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8568c12e9f1a444d0914f64386f0190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c69efc00629a6fa23d714fdb6e006e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ac79f51688468978dfb02ba6eed85997(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 288, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de275957176cabe944712ce774e4dedc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac79f51688468978dfb02ba6eed85997
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_c77dfd2fc4b3740c8b7dde1627a6cbfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df9877dc51d8530c530af48a43e78dcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83ed8bc83fffccfcee982bd71f84cdfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d636a839c1554b6e0420f4abb46f545f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83ed8bc83fffccfcee982bd71f84cdfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d636a839c1554b6e0420f4abb46f545f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83ed8bc83fffccfcee982bd71f84cdfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d636a839c1554b6e0420f4abb46f545f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83ed8bc83fffccfcee982bd71f84cdfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d636a839c1554b6e0420f4abb46f545f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83ed8bc83fffccfcee982bd71f84cdfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d636a839c1554b6e0420f4abb46f545f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_990786bb38024e86bf09b9526a4019cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23efb7705a00d013dde414a35a50893b
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_21ec3457d5a65059159dd1b636a87f8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c0fe7ab31e463748f80f95818a23727
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 256, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_19f33c7b3c45ba488a6506ae11a7c12a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83c3cdb6e829f210bfd695bbc50ad272
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_19f33c7b3c45ba488a6506ae11a7c12a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83c3cdb6e829f210bfd695bbc50ad272
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_19f33c7b3c45ba488a6506ae11a7c12a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83c3cdb6e829f210bfd695bbc50ad272
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_19f33c7b3c45ba488a6506ae11a7c12a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83c3cdb6e829f210bfd695bbc50ad272
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_af60ed4ac0b186349e071c806ab104af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf95189c0e8c270fd0d242cc8a5f9adc
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.33445578813552856], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e0242c1fa2029637fa51c2ec6c7c9569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7996d0f16b68a00175ac2ee58b32986a
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_b108fbc39256bc5b9aeb8766307a0963(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_425de543daf868ba0cce0b31447a73ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_c0d6fe9722f0799fbb405abe0270e336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d22fc4b89bad4a9003a61e438ce2bf4
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0d6fe9722f0799fbb405abe0270e336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d22fc4b89bad4a9003a61e438ce2bf4
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c0d6fe9722f0799fbb405abe0270e336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d22fc4b89bad4a9003a61e438ce2bf4
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b92b13d27e3aa19ce00b8d592d5cb336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6e35b8f487625603ffedcfab9df3688
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1a748b009e681bf2b8308e75a7caa464(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b2456f79ae510f435d8c854742cf134
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_17696e5af717e1533cedbc1d01606ee9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b01629033fc9528fcc55a5bfc15c4fc2
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fd762c607a9b571472f0a199a47b92c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1174, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b2622e2f4e7be368dc16290e6d3443ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd762c607a9b571472f0a199a47b92c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1174, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_efcafc96950dcf129fe59747fbcf9793(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1174, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc4bff8438bf16bca2dbb7461b169d74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efcafc96950dcf129fe59747fbcf9793
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1174, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_37d4d0462f0f172a96e81db797d54de8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4029ddc8f57d7ecaa7a07ae059eddc47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37d4d0462f0f172a96e81db797d54de8
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b54fa9926393e0acd0654965f97e03e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad4d8b2af8f1e48e6c666ef31b8c2059
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 1, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_80af52dfe292d72107a04668b0cd1343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5328717a8cee22cfcdd245b743fb930d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_aec90cee400750fc01876ab89de1e4d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffabbc4f4d7bfe414d05eb9696e61e7a
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 44, 44], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_f9955a38bee4577544ae8d824991b4ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce4ace322016d97bf695286595d16834
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.006607949733734131], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_cc449860d6dd09ca08315205070b0d5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d835d8df3c2e970a3a833e74afa70962
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.029899924993515015], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_d933ddda070bbaef8e5f1c59ff64cae0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_911be43c55a1c2e24ab4cb563f65e439
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_e5f83a1ca6bd07617247d9db08a03faa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acb75ac7d8898f56f550b9ebe95dc4b7
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_e27b4de0cc9dcbf2abd876033a414f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_094c190833ee8d62c1961fdea27f1fae
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e27b4de0cc9dcbf2abd876033a414f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_094c190833ee8d62c1961fdea27f1fae
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e27b4de0cc9dcbf2abd876033a414f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_094c190833ee8d62c1961fdea27f1fae
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e27b4de0cc9dcbf2abd876033a414f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_094c190833ee8d62c1961fdea27f1fae
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_4662e8fc3a72131054e383679e495cbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eee649416a3696b12166505bb1d12aa7
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([950], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_c8fd2afaca49d8e4d1d824257eaff2da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89e3fd03c0dad8dcdb4883f123d86317
    def get_inputs(self):
        return [
            paddle.uniform([145, 480, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b31fe694663a63c77eaf9110c02771f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79fab2ddab9377dd34e70e31aa78037e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_4aba0d158105d282ffba17bed53707bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5e7aa5d68a57e6474e9d1052ed0fe1e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d79e6833d3f91e08207755f2dfeb43d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5328717a8cee22cfcdd245b743fb930d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_1048cbcfad1e57733060726eb1f9eb80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66b314de6dbc0fb9bda9195c20505f5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ec02483c8f9db3dd29d1e78b1f2bc79b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac79f51688468978dfb02ba6eed85997
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_27c612e346896d233ff9bd7b246d49d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de528f7e6e64b9d54e1cde1c5384933d
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3086425fc951090c34f0bf80c1f10896(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c514e2a5c05418dc0e1b53b10e40f6b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_c084c1131988b4db2caf693eadf09142(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5067f5666ed30cfdbbb90847d8b3066a
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d725ad75482a80a14e59b3b3c00fc205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f75c0dd4434232b4fd14b27654a58105
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_8a5c77c474720a9a27675d3476c5b2e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d70b658edeaf15347771e0b56340e195
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b2d4d7a3a22170d040d1a37b98cbb112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a64fa566b317e48c6abbc2182f3c730e
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e95b3686b428e6bdae6804fa65bffa1e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 198, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[86, 198, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_91dd385d16e3a2eb497a96694c1d4653(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e95b3686b428e6bdae6804fa65bffa1e
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 198, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5dbddaa920da02509d65b7dd39dc68ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 198, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[86, 198, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a4cf3915059997951b7a41bd31e281ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dbddaa920da02509d65b7dd39dc68ab
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 198, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9ac0573c952ebf3d505fcc06504b2671(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 198, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e74a6002e126cc97f1cd3aa381e761f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ac0573c952ebf3d505fcc06504b2671
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_65cab4d44bf19c05456e48136257c334(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f50576135ce73f574a86b68017ca20fe
    def get_inputs(self):
        return [
            paddle.uniform([10, 36, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 36, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_322aa25f58dfdf3f0c0d9727256945ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f75c0dd4434232b4fd14b27654a58105
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d38bdb5e14c21f8bb9d48e9cc873a71e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea3ad6a4af220e8de872eaa59cebe711
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_84ff80f8752e2e57edd1b5c325eb57ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88e51f2bae392ed7d4c6b58929ef6a83
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_939acae69183dfe6065e086668414093(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d9410094fe9637e3e6b73888ac26824
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 15, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc219cf526e458de33df21ff9de38ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382b20b4b9f61800ddbe43e1cf3aa712
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_e4f9580614cf2b59933a55b3ff599022(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b036567a61f9ed516a526e013a1b778c
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c8cd99e444dad3dc78e88ca0a51adf00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab75d6f916b8c515969c8108422557d3
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d6172caada93e27ce0f496ad3beace0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_597a6361be11b23fe2d262c45726cc7a
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_27616106a6dd86d5554fe1a1875bd07d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c54270e10cb3f0d4753cec8cd8b9ecda
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_6b621a0d398b2e8d1cce1f1970853587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d066935a2d690803217a9d50515ac25a
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([247], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b2708fa5f4a470d14b8940ff27a34aae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37d431d8d12ac49415bb9099f8f1b721
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d725ad75482a80a14e59b3b3c00fc205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f75c0dd4434232b4fd14b27654a58105
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_6c744913355180c901b05db58bfcba3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ab8a4edf80a48e3437fcca6a1eb9987
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d44cc452609a27876d90c8d93b6e990e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06afa0ac8681719902f3e86d0feb1b9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.4614502489566803]], [[0.4676744341850281]], [[0.511082649230957]], [[0.21530066430568695]], [[0.4868157207965851]], [[0.5041162967681885]], [[0.328726202249527]], [[0.5048556327819824]], [[0.33760979771614075]], [[0.4009786546230316]], [[0.43608835339546204]], [[0.5867417454719543]], [[0.5625014901161194]], [[0.36064696311950684]], [[0.5384686589241028]], [[0.5432965159416199]], [[0.41655802726745605]], [[0.40423348546028137]], [[0.4430371820926666]], [[0.5660258531570435]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_e8d878ebb439c20b2ae99967473a9d37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08a35fac70f021394b01b6711933f04f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_13bf0e2e5799f4d87f6e27cb6ed912cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34967eb0bb98058b9f8a9a8cc0d2f7e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_975f0b7d4080666d88f8ec9abf47c529(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ae2202f46a54d568740a3992dd4d5a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_d425546b12196b2c74feff416b405932(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec9ee9d1e6d1914b48ea8d88287b037c
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d425546b12196b2c74feff416b405932(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec9ee9d1e6d1914b48ea8d88287b037c
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d425546b12196b2c74feff416b405932(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec9ee9d1e6d1914b48ea8d88287b037c
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d17b3eda186f14a705c5a4930d1297b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 9216, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 9216, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_80f305e057021fb95b8a1e9428af6b80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d17b3eda186f14a705c5a4930d1297b7
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6b26322edd22e347c23b13c8c14f672b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 9216, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 9216, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b83018ec9efa707eb1ea4398f1d9231(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b26322edd22e347c23b13c8c14f672b
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 9216, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_4be8869c253a0f11a062831a46528ee0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 9216, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4bf47e931f3ea79362691880a6b8f823(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4be8869c253a0f11a062831a46528ee0
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c6eb6d368fe7d2580c533c5cc2e8e67f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 96, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a538b802eff7b2de7191c8eb18d32526(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6eb6d368fe7d2580c533c5cc2e8e67f
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_c80a6833fc1541cc7b0210a24432a25c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d6d9eb946c5ddbd8b2a7896ee540eec
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a538b802eff7b2de7191c8eb18d32526(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6eb6d368fe7d2580c533c5cc2e8e67f
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_efb0f55382161727fd74a31d682e7a6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_934c25cb1f28f1284c4a62c7a4fddccf
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a538b802eff7b2de7191c8eb18d32526(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6eb6d368fe7d2580c533c5cc2e8e67f
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_de63c2d17a68a535510e4d6254ac5115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_564aa4804d2ab58f4f8afc374ce66739
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 120, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a538b802eff7b2de7191c8eb18d32526(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6eb6d368fe7d2580c533c5cc2e8e67f
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_4a3ed92cb47a4e38882f5b0cc2f0c875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcedc90d2624fedf3a4962d18492559f
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 240, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9ee9cf7fc3a358e6d76aa0f211746e1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 24, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_060139578dfe3bcb198aade1f741343c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ee9cf7fc3a358e6d76aa0f211746e1b
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_ed69f1e533d87a7adad18232b1a5bfe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c7f864af7eaaeefb609476efe0becd5
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_060139578dfe3bcb198aade1f741343c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ee9cf7fc3a358e6d76aa0f211746e1b
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_29a681639f03738b8d2aea5ec80f8343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_691d6d737f0d01cb0a668e4c3cfb1df0
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_060139578dfe3bcb198aade1f741343c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ee9cf7fc3a358e6d76aa0f211746e1b
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_d11c2b95447a650943c2a0026790350e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ffe346411e55f42fdce51b4b4d9aeeb
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 120, 120], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_060139578dfe3bcb198aade1f741343c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ee9cf7fc3a358e6d76aa0f211746e1b
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_30d8a7a995321090517840815a0b4951(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18473e19dd47ed997b04632d88ab08e8
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 240, 240], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ec02483c8f9db3dd29d1e78b1f2bc79b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac79f51688468978dfb02ba6eed85997
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_0cb4117daccca0cc59711579b85b0837(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5637e340af3332d7784201c0c63c067c
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_36363d7e097a47568edd8771d4a9d17a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2481e105c4881796aef69e0dd25a9d25
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_38b9310cc0867674dde56ea03f1640a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4512e33d194e9bbafd5ee55e70014114
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3086425fc951090c34f0bf80c1f10896(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c514e2a5c05418dc0e1b53b10e40f6b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_55d475a663ed8939ac599a267ca5a6c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a2bb000481d11820bde8e63e87f442c
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 17, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc219cf526e458de33df21ff9de38ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382b20b4b9f61800ddbe43e1cf3aa712
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_b63812cc67d6d63e39b8d8d163958991(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9eaf54001e5dd18b6a49a167eb27dc6c
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e0242c1fa2029637fa51c2ec6c7c9569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7996d0f16b68a00175ac2ee58b32986a
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_e5610bed6cc811f77c6fd06d318d4da8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9f741b9eec58ea6d971c878a6899c5
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_cbeb641ef7859ada24e835241bbc3fed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d3890b2dd49a51fa89038161d787d38
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.25921350717544556], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_0cea6572670fc2f649017c3999918d29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22a4d1373b2bac1f63849563cdaff296
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc219cf526e458de33df21ff9de38ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382b20b4b9f61800ddbe43e1cf3aa712
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_8d160ffc933afa511db06fc6fa86323c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48c1f0b3aa5a6b888dacff0000fcac2d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d2f232ee100ddc99ba29b4566971f559(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bdabd9a9d34cad48e35cda77addf8037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f232ee100ddc99ba29b4566971f559
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_4751e898c29391a9ddb80261f8cbd24f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06e3aeaa24fef692fbebc0f3db6bdcf8
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_58403540bae4aae8ef485462f78a781c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed1ba3c28e87e79667627287243f3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58403540bae4aae8ef485462f78a781c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed1ba3c28e87e79667627287243f3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58403540bae4aae8ef485462f78a781c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed1ba3c28e87e79667627287243f3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58403540bae4aae8ef485462f78a781c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed1ba3c28e87e79667627287243f3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58403540bae4aae8ef485462f78a781c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed1ba3c28e87e79667627287243f3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_85b6b9dd37b1d80bdb8af958728d051d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97eab3f01facdd16e7808d2e9a26d546
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_85b6b9dd37b1d80bdb8af958728d051d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97eab3f01facdd16e7808d2e9a26d546
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_85b6b9dd37b1d80bdb8af958728d051d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97eab3f01facdd16e7808d2e9a26d546
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_85b6b9dd37b1d80bdb8af958728d051d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97eab3f01facdd16e7808d2e9a26d546
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_85b6b9dd37b1d80bdb8af958728d051d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97eab3f01facdd16e7808d2e9a26d546
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_b3a40839794a50c5411765d1eae0c3bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08cb7f26b76ddd9a4c61c34b99be7b82
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.22468024492263794], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_81d127e66eeaf3a06d2a06fdf9e5c504(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dd8f5115e9b9b191e1a05315edd9c17
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_44358fb7fc2635ba4bc870e83cfce997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4b66f6662af9bd1d97588954ac2d0fc
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 44, 44], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f3d932ca3fab708661f9d8a4131a18f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1571de3b7504056e385f67f78d5a62f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_2ae8a9c4960c11d0a4d42139fd28e20f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2dbec6386880e81321107620b7579f23
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_f3a93c12dc6957319613633a2321c8ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f3a704bbef93b0703c705ca2a8835d2
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bcb9873187d5fe3989abe828731bb5fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4245913426c50e23e26e996c4e3e55d4
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_6f62a269e88a9b53d8d6ef3b7ceb3fef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0b9f03b1162d46c3768e32b17f3c7dd
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.47522804141044617]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.07023847103118896]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_fd3bfc3a1188e9a31e5afd28ba48be21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0b9f03b1162d46c3768e32b17f3c7dd
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.7297652959823608]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0865829586982727]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_cdb50ba0c8fd0eb784c36f8631b94f7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0b9f03b1162d46c3768e32b17f3c7dd
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.47522804141044617]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.3959505259990692]], dtype='float32').reshape([1, 1]),
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


class TestPrimitiveOp_a2138526c5f6e99a17235014fb20e663(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d467e8235e880f3c5a8f5baf9bacd88
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.4047892391681671], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_78369d35a023e766a83c9ea6b0cd0056(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97f46f2ddc769f1123ff6955cd790a0e
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f93f9e1ddf87342fd8d308ae94592522(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7018c8290aea098c5a3f2ac31a3962e
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_457c7db8651fc3c3a46615b2680f5cca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_732a0463433346915e9e89daba81721a
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7d136372223a8a15f4c6a86a1d669690(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51c40fa1586b97a36bcde0f702df97b1
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_def03e62a4814d783b71ad29c5d28820(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7863b2beff958c4ac56a444880cee21e
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_331180ea98e0a80a458572ff2acd765a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6703f810822211bca67b8219701506ea
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9148ad62922d4d07437296be051718f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32ae02440098c63baf59fb574aae1c13
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9148ad62922d4d07437296be051718f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32ae02440098c63baf59fb574aae1c13
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9148ad62922d4d07437296be051718f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32ae02440098c63baf59fb574aae1c13
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fc6110d442cbfba8294c02b73b7062cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1d9e83efcbd2dca805b623e367850025(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc6110d442cbfba8294c02b73b7062cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aaac6ea7ca2fbda99923540dcb8beb16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8a7b82d462f2107ba9f271eabb11776
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_09ea9eff0dbc5c5118e099bec359063f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcedb87a3d86cde827f037ac3eea584e
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 50, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e25053dd9a75fe59c592df9e1a00251(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3b25d945a95df602c53bcc90fe0b3be
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f3d932ca3fab708661f9d8a4131a18f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1571de3b7504056e385f67f78d5a62f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_fd4489de0ff0c6977599d14b061fd74d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_baeca77b05250f7e43ddf14d51ae9565
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_972c6355a3057686f4f2269153927742(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f773f57b350be988ac85f55a5df29df3
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_80af52dfe292d72107a04668b0cd1343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5328717a8cee22cfcdd245b743fb930d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_3106412923dce46b3894dcff9bfc71b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50a61ad11597cfa1516622801b49ecda
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_7004194903d8cf9f1a508a4c35911118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f4d4f7e433464980971678b4bc8860d
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_85b6b9dd37b1d80bdb8af958728d051d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97eab3f01facdd16e7808d2e9a26d546
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_85b6b9dd37b1d80bdb8af958728d051d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97eab3f01facdd16e7808d2e9a26d546
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_85b6b9dd37b1d80bdb8af958728d051d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97eab3f01facdd16e7808d2e9a26d546
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_85b6b9dd37b1d80bdb8af958728d051d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97eab3f01facdd16e7808d2e9a26d546
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b92b13d27e3aa19ce00b8d592d5cb336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6e35b8f487625603ffedcfab9df3688
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1a748b009e681bf2b8308e75a7caa464(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b2456f79ae510f435d8c854742cf134
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_17696e5af717e1533cedbc1d01606ee9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b01629033fc9528fcc55a5bfc15c4fc2
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_990786bb38024e86bf09b9526a4019cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23efb7705a00d013dde414a35a50893b
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_167278cf444670d125f4eda61c902d4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_903b96e71dcc2edb4e7ec677c1e6f7d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_91b4a085617e7da2def34cf8d6c74956(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 576, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 576, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f87b6fae9c2a27b487b96660d8ce1c12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91b4a085617e7da2def34cf8d6c74956
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 576, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5249268acc5d17b84ac68e232131d280(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 576, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 576, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cd2e8f9fdcad7be8972065f429a33ec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5249268acc5d17b84ac68e232131d280
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 576, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_17cef0ee270a1d6a2d5b3a9a5d931ec3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 576, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a09dcade74eef5a909f8192066a1eecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17cef0ee270a1d6a2d5b3a9a5d931ec3
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d933ddda070bbaef8e5f1c59ff64cae0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_911be43c55a1c2e24ab4cb563f65e439
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_20e4dd2d31f4f5361e003826ebe35216(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1025, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1780b5276e7509f691c575d8b42731cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e4dd2d31f4f5361e003826ebe35216
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1025, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_21179955b4f9f0e3ceb9beecb1ffa18a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1025, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_735f97ec88649035f456e3dfd86daefd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21179955b4f9f0e3ceb9beecb1ffa18a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1025, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c561e11f1a0e263725086457ea291273(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1025, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6caabb53cdbfcd40563df25dd9ec72fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c561e11f1a0e263725086457ea291273
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bdabd9a9d34cad48e35cda77addf8037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f232ee100ddc99ba29b4566971f559
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bdabd9a9d34cad48e35cda77addf8037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f232ee100ddc99ba29b4566971f559
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bdabd9a9d34cad48e35cda77addf8037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2f232ee100ddc99ba29b4566971f559
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7ae457acfcce02b7fa929e0b3a4c4c15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0431d17b2c9917e69e59fceae99f1607(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ae457acfcce02b7fa929e0b3a4c4c15
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8b52c36df5d9cb80da71f086e5d714df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c53df8392ce8625ae243f2ee7b563a5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8b52c36df5d9cb80da71f086e5d714df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c53df8392ce8625ae243f2ee7b563a5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8b52c36df5d9cb80da71f086e5d714df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c53df8392ce8625ae243f2ee7b563a5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8b52c36df5d9cb80da71f086e5d714df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c53df8392ce8625ae243f2ee7b563a5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ec02483c8f9db3dd29d1e78b1f2bc79b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac79f51688468978dfb02ba6eed85997
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_972c6355a3057686f4f2269153927742(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f773f57b350be988ac85f55a5df29df3
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4fe93ab0db01dd2a2aaec1c7bbe3ddcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c19d801d8a9adac0c877ec38bfa79311
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_9db0116170740a72b05a802b7725e487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5b4acd9abd9ffb99bb81463fec102cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 84, 84], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_e828c4a1b2e6a72be3ebe6027d73ebb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8bc4720b1c97bc36613c2f8acca09784
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([70], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc219cf526e458de33df21ff9de38ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382b20b4b9f61800ddbe43e1cf3aa712
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_f67ecea72fca8134a053a35d57e26a44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d760e43b8bcab16342b5849d4aa01b62
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_cb3a741614052b0105b589e147f74a52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9afbcc6f57ba50a3da6628142dd38ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.15759488940238953], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_80af52dfe292d72107a04668b0cd1343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5328717a8cee22cfcdd245b743fb930d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_856bef5ee38d4ee1cc041989f4071e20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db8033965d9a3d28afc98b44727dd707
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a5c3c3c06a0d13b96cdf431c9ceb62f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ecfa147eb81fe4e6c4ac0a05c00bd6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a5c3c3c06a0d13b96cdf431c9ceb62f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ecfa147eb81fe4e6c4ac0a05c00bd6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a5c3c3c06a0d13b96cdf431c9ceb62f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ecfa147eb81fe4e6c4ac0a05c00bd6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a5c3c3c06a0d13b96cdf431c9ceb62f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ecfa147eb81fe4e6c4ac0a05c00bd6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0fde7c304ac49b13618504d8a6c1e81e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf6ae6ee44a92275d56a63ee17bb8b8e
    def get_inputs(self):
        return [
            paddle.to_tensor([5498], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_82fd8c704b8b8c6542c9baf3ca28aca8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4684257e4dc9bcf2c6b7561728450da5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4684257e4dc9bcf2c6b7561728450da5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4684257e4dc9bcf2c6b7561728450da5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4684257e4dc9bcf2c6b7561728450da5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_419dda6513413fc4400521e636384524(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([5498, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_419dda6513413fc4400521e636384524(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([5498, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4684257e4dc9bcf2c6b7561728450da5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([5498, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_81d127e66eeaf3a06d2a06fdf9e5c504(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dd8f5115e9b9b191e1a05315edd9c17
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_11285b172637ac41d5c24859fa3716c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d4f4b41a1f8e98e927866cbdb496f4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 76, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_322aa25f58dfdf3f0c0d9727256945ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f75c0dd4434232b4fd14b27654a58105
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_f2291ae025bdb04beaecf84f95e0804a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_513ae1739a28cafbf5128434a2ca254b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 42, 42], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e76985a4e90f64c2bd2027e7ac3541fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bf61eef8fe289dfd9d842a35208591
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e76985a4e90f64c2bd2027e7ac3541fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bf61eef8fe289dfd9d842a35208591
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e76985a4e90f64c2bd2027e7ac3541fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bf61eef8fe289dfd9d842a35208591
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e76985a4e90f64c2bd2027e7ac3541fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bf61eef8fe289dfd9d842a35208591
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e76985a4e90f64c2bd2027e7ac3541fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bf61eef8fe289dfd9d842a35208591
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_8fce38251fce1cc940e99a6090b7f91b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7138806c53fe65047584c00ade942317
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.1235460638999939], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d725ad75482a80a14e59b3b3c00fc205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f75c0dd4434232b4fd14b27654a58105
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_58d4195354b6c7d74f461f2d942405f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13831282a9b56e6317826d2bfd8f8fb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cebe559d7af21eb4b81f8dbd9110f87e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c514e2a5c05418dc0e1b53b10e40f6b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_18c8ea28ed5f019dabcf835b29a04d02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2f5611870d3f1d8af810feb55e461d
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_d316cf13ade969b6b3d15db1115df80e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a456f0aa8c14d027aa36c5d72198f1a
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


class TestPrimitiveOp_763f1202ae6ad71674911e33c8dd3168(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fb038135fe2fa8372d9386ebb79df7d
    def get_inputs(self):
        return [
            paddle.uniform([11, 112, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


class TestPrimitiveOp_58403540bae4aae8ef485462f78a781c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed1ba3c28e87e79667627287243f3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58403540bae4aae8ef485462f78a781c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed1ba3c28e87e79667627287243f3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58403540bae4aae8ef485462f78a781c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed1ba3c28e87e79667627287243f3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58403540bae4aae8ef485462f78a781c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ed1ba3c28e87e79667627287243f3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be0635f86a7e3274c04c58028396ed3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf6ae6ee44a92275d56a63ee17bb8b8e
    def get_inputs(self):
        return [
            paddle.to_tensor([1074], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_b6b96021ce35fc86286c25e66e3ae83f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_79334c1cc80d1092fd19075c8895e841(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_79334c1cc80d1092fd19075c8895e841(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_79334c1cc80d1092fd19075c8895e841(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_79334c1cc80d1092fd19075c8895e841(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_97c20437b5de1f7bffe1da7aa3a9519f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1074, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_97c20437b5de1f7bffe1da7aa3a9519f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1074, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_79334c1cc80d1092fd19075c8895e841(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1074, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e48853b152277be664fdf715e22fce81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61974deeb2d3a4aa863bbf228e52e4dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_7885593c71590d2b86ad6ecc4c29827e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_448cf51c0a1fffe56b88bcff73ae7eda
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9a6a36e2e47e9e78e5ca0f5a6a8e3618(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_622ca4fce08683dab2491438ceb8c2d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_95d554e985c6f48cc12dbe7c8f1794f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eb265886210cdcdffd9bb7c69061d80
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_39c493482373db37b12ad1412d08f9d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123d129ef1079efd3b36a7c6710f06e0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f3d932ca3fab708661f9d8a4131a18f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1571de3b7504056e385f67f78d5a62f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_ff4f675465150660b91273c179bdaced(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3e2ef532b1395ffc405f5306b0727a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_86bae15ea8e123e080709e45e5a972b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4681047ff92aef0deec628951357c2a2
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9fd630d22fa1284e6cf55fa9657803c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60a6a9d5d73b03004e18a33ade3fb789
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fd13547b0e011482049eabd6c38ee327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e47583a69e57b4d722fda1acf601cd94
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_59a3185ecb207b8a35e34c0b35510afb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a6ac6e417aa2f9e70b7543858833ed7
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_238a268aa86592de626adffeb0d5737f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 97, 97], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_783f9101b3e9f8267b0dadb38f67e4a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_238a268aa86592de626adffeb0d5737f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_424ebfd76212c1da86bd7d09a130e2eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf6ae6ee44a92275d56a63ee17bb8b8e
    def get_inputs(self):
        return [
            paddle.to_tensor([1773], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_9f8ebb06a6bf07d375f1b7e9fae6a07d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f13614cc2537db6f367073b58b4fb38d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f13614cc2537db6f367073b58b4fb38d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f13614cc2537db6f367073b58b4fb38d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f13614cc2537db6f367073b58b4fb38d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_34d5dda64574ae72acebdf0c977f4885(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1773, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_34d5dda64574ae72acebdf0c977f4885(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([1773, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f13614cc2537db6f367073b58b4fb38d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1773, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_322aa25f58dfdf3f0c0d9727256945ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f75c0dd4434232b4fd14b27654a58105
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_3f80c33f13528440f791250ddc9c66d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8098960eb115a79955652f745c57d4f2
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 22, 22], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_d7199bb25eb0b7f73c31364686f76dfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2080653d0e0c3645fa313a40c82a396b
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_820df3ac1c04e90bc8826e6610e9140d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 197, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[86, 197, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7f3947240f38187ad8fd5e7ca36daf34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_820df3ac1c04e90bc8826e6610e9140d
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 197, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ad45ac7bc40b2fae29b5363c664de0dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 197, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[86, 197, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cc658cf738e59eb7805ba35037aec1d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad45ac7bc40b2fae29b5363c664de0dd
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([86, 197, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_875c66c073a0c9711221b6128299efe6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[86, 197, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_18804659a32e5c4abde5abf323365569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_875c66c073a0c9711221b6128299efe6
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_33d29b4b280b656c19117380d97cf363(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea0be2fd07a5b31350bc3ce58eceaa31
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3177f9dd01ebc8e21c5a191dbebd0630(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46f0e721caa6ba927b4d65e64edf8ad6
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3177f9dd01ebc8e21c5a191dbebd0630(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46f0e721caa6ba927b4d65e64edf8ad6
    def get_inputs(self):
        return [
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8407e2d8d2f2d788039899ada63862c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba5cc92704ec940eeadc35b0cfa3e2ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_8cdbf0a1c6ea4a48d2491a6c381a039c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_432c37706dfe1d909f11741a7efb46a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bba7aec2e29072e67649ba96732e6783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f714bbf021d42dc1978a94f2e6726786
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc219cf526e458de33df21ff9de38ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382b20b4b9f61800ddbe43e1cf3aa712
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_1533f6a709f2c1812bf7e0646994cd58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_340311d4fd3d640fd702110a039dfe0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 17, 17], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a5c0916297b8397cb48b484b07f7ef9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32074b59bac5abaf1b8a9d10910ce304
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a5c0916297b8397cb48b484b07f7ef9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32074b59bac5abaf1b8a9d10910ce304
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a5c0916297b8397cb48b484b07f7ef9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32074b59bac5abaf1b8a9d10910ce304
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a5c0916297b8397cb48b484b07f7ef9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32074b59bac5abaf1b8a9d10910ce304
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a5c0916297b8397cb48b484b07f7ef9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32074b59bac5abaf1b8a9d10910ce304
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_d8b8943a6c813581ad59ce2e0f0910c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d651b3382091c8333b6aeb3076a83d6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.48771584033966064], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_5c874007b662b58d267a03ebe34c108f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd3206ed2fe7ac43381c2bbb0e5d2f9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 10, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.2867005467414856], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_d45ba9b26809ee86adf3dadcc6e03895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7541fd6af8509f0ce8734aa60e436973
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_544980cefa9e1648b42804fd856c1fc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e627e1149ad0833fb50dc5f6837362b
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_1f051fb66aace7394166f242916ffdd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49df513be1b8d3a4da5292a4568c65d1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.2099647521972656, 1.902867078781128, 2.406503200531006, 1.7131986618041992, 2.461681842803955, 2.3766579627990723, 2.1742782592773438, 1.6286232471466064, 1.5602257251739502, 2.41302490234375, 1.6441746950149536, 2.416710376739502, 1.8515286445617676, 2.1799139976501465, 1.9803026914596558, 1.7719007730484009, 2.564967393875122, 2.3181722164154053, 2.7134108543395996, 2.3339548110961914, 2.5062081813812256, 1.9509363174438477, 2.308413505554199, 2.0323667526245117], dtype='float32').reshape([24]),
            paddle.to_tensor([1.489646553993225, 1.160320520401001, 0.5958489179611206, 1.277940273284912, 1.1370296478271484, 0.9593456983566284, 0.9152644276618958, 1.1150310039520264, 0.8198758959770203, 0.772907555103302, 1.0535266399383545, 0.9154517650604248, 0.7009066939353943, 0.5507380962371826, 0.5811593532562256, 1.3856992721557617, 1.4507875442504883, 1.0309391021728516, 0.9635930061340332, 0.926554262638092, 0.6079788208007812, 0.9355449676513672, 0.9881537556648254, 1.358212947845459], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_05c54dd88fc737bffa858c40c9fd294a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49df513be1b8d3a4da5292a4568c65d1
    def get_inputs(self):
        return [
            paddle.to_tensor([2.3055405616760254, 1.8568456172943115, 2.397026538848877, 2.2314541339874268, 1.971043586730957, 1.6924320459365845, 1.9586052894592285, 2.045534610748291, 2.4093687534332275, 2.216003894805908, 1.8692476749420166, 2.13918399810791, 1.8258346319198608, 1.908278226852417, 1.9066638946533203, 2.1788673400878906, 1.9698405265808105, 2.2014951705932617, 2.1474692821502686, 1.6641182899475098, 1.8911681175231934, 2.3652284145355225, 2.126494884490967, 2.16676664352417], dtype='float32').reshape([24]),
            paddle.to_tensor([-0.4896465539932251, -0.16032052040100098, 0.4041510820388794, -0.27794021368026733, -0.13702961802482605, 0.04065430164337158, 0.08473557233810425, -0.11503106355667114, 0.18012410402297974, 0.227092444896698, -0.05352669954299927, 0.0845482349395752, 0.2990933060646057, 0.4492619037628174, 0.4188406467437744, -0.3856993019580841, -0.4507875144481659, -0.030939042568206787, 0.0364069938659668, 0.07344573736190796, 0.39202117919921875, 0.06445503234863281, 0.01184624433517456, -0.358212947845459], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_fcce173eaaaaa2be953867095cea9e05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49df513be1b8d3a4da5292a4568c65d1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5407916307449341, 0.477561354637146, 0.6006683111190796, 0.3922886848449707, 0.6322284936904907, 0.5872103571891785, 0.5390008091926575, 0.3951663374900818, 0.42829421162605286, 0.5920706987380981, 0.40803179144859314, 0.5983115434646606, 0.46096092462539673, 0.5144696235656738, 0.4873649477958679, 0.403733491897583, 0.7083107829093933, 0.5804455876350403, 0.6732016801834106, 0.5711895227432251, 0.566274881362915, 0.4944098889827728, 0.576564610004425, 0.4960557818412781], dtype='float32').reshape([24]),
            paddle.to_tensor([-0.22710701823234558, -0.028592228889465332, -0.40487539768218994, -0.1511639654636383, -0.34464552998542786, 0.04285043478012085, 0.20903432369232178, 0.292479932308197, -0.4828256368637085, -0.33061206340789795, -0.30776816606521606, 0.4009280204772949, -0.14245843887329102, -0.10920903086662292, 0.4103533625602722, 0.46145617961883545, 0.3521844148635864, 0.3356350064277649, -0.3771110773086548, 0.24811875820159912, 0.44464045763015747, 0.46116816997528076, 0.1386188268661499, 0.42249423265457153], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_40fbd6080b85fe32b9d63842ca51d8a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_391ea07d377bd44a525315beafa2d798
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 1, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd9fdc6138d4d9f8358cb658b65e3070(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67448f2337e9c3d3bcfd73b04524295f
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_80af52dfe292d72107a04668b0cd1343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5328717a8cee22cfcdd245b743fb930d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_cd958e51adc1c2d802656e406fbf4302(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da5da8a21c60e75fee94a83a9ddcb386
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5f3ba540038cb96444f3902ba8abbfb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0399b6b5890bd2e28b6c4105967fdbad
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.23405253887176514], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f67ecea72fca8134a053a35d57e26a44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d760e43b8bcab16342b5849d4aa01b62
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c3693505b281b397a50554a882deca00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([17524], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([17524], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c3693505b281b397a50554a882deca00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7352dd94434211fb041a08a13babfb1
    def get_inputs(self):
        return [
            paddle.uniform([17524], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([17524], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d79e6833d3f91e08207755f2dfeb43d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5328717a8cee22cfcdd245b743fb930d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_a79c84b757ad546355bdc0fca14cc4c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edab10024c1371104d300a259348b016
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 24, 24], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_4e7170fa7f1b31fec00ad32cbc90b5d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0c7be1670664e593a2faa44e7eedefd
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_e83b769e13c45039427c0db983a8eb57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc7f49a71d68be5f48093de4be1538c8
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 1, 9, 112, 112], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_baed70d442803c3e98b03d9699acfab9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b24ad60e80ab24bb6b066b48623adab2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_baed70d442803c3e98b03d9699acfab9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b24ad60e80ab24bb6b066b48623adab2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_baed70d442803c3e98b03d9699acfab9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b24ad60e80ab24bb6b066b48623adab2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_baed70d442803c3e98b03d9699acfab9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b24ad60e80ab24bb6b066b48623adab2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_baed70d442803c3e98b03d9699acfab9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b24ad60e80ab24bb6b066b48623adab2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_53670a4f9a46fe2517e0112981f6cd58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_509dc620ff46007c0b8a2e4224bed677
    def get_inputs(self):
        return [
            paddle.uniform([950, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([950, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_53670a4f9a46fe2517e0112981f6cd58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_509dc620ff46007c0b8a2e4224bed677
    def get_inputs(self):
        return [
            paddle.uniform([950, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([950, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1855619832ac99eba8348e43ad634843(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[950, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1d0fc574f628376dfa0a3c9f8cacebde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1855619832ac99eba8348e43ad634843
    def get_inputs(self):
        return [
            paddle.to_tensor(-1.0, dtype='float32').reshape([]),
            paddle.uniform([950, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_53670a4f9a46fe2517e0112981f6cd58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_509dc620ff46007c0b8a2e4224bed677
    def get_inputs(self):
        return [
            paddle.uniform([950, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([950, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9858fe36835d0cefdfd759d65464119c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91dc404a62a9ef53dc011df69dc0346f
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fd19463235a844464ea87852f3244804(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 577, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 577, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_42cf946fed67bbfe24b49f77f015fe71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd19463235a844464ea87852f3244804
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 577, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b0a83136171cb2d5466391cd327b76f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 577, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 577, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5ebf616f179c6a7d940dc138e8f6ace3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0a83136171cb2d5466391cd327b76f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 577, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_03649308651dbda05d4b034783a5651b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 577, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e21793b012577b29faf54a7947c416c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03649308651dbda05d4b034783a5651b
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_f93c23ea1698adb309c0e7762bbb1228(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61809abe8e2d7d7fc05391c33fd36811
    def get_inputs(self):
        return [
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 577, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_604ffdd8a2325ef4c6daaf65dd515e55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5b8478af74dca2c2504ea117ed35028
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 1, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_b60c316429c4a6fb7024ed40a193a6e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b07049406cfd624569bc2827f1dac0b
    def get_inputs(self):
        return [
            paddle.uniform([1, 156, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 156, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2eb1af84e0b1a761cafa4073ef8f1cba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3df821dbf04205963b7a5e9accd1888f
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_6635f886765db961cbebe1197b393293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88266f0a88e56578e8a70fa0f3a56d8c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_5d43531c2930564db13292655e26c7cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76285b4e18b0326c79be4007407e9750
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([2204], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b92b13d27e3aa19ce00b8d592d5cb336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6e35b8f487625603ffedcfab9df3688
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1a748b009e681bf2b8308e75a7caa464(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b2456f79ae510f435d8c854742cf134
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_17696e5af717e1533cedbc1d01606ee9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b01629033fc9528fcc55a5bfc15c4fc2
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5b3deac9630749ef5b9e9e32d208444b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08cb7f26b76ddd9a4c61c34b99be7b82
    def get_inputs(self):
        return [
            paddle.to_tensor([0.015151619911193848], dtype='float32').reshape([1]),
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ade6949436cb45ff5f454da6a9cc0d9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1d72b749c9e238b5f05268b163789e1
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.26577940583229065]], [[-0.048389434814453125]], [[0.174646258354187]], [[0.2115311473608017]], [[-0.48888450860977173]], [[0.5408591032028198]], [[0.06501791626214981]], [[-0.4082222282886505]], [[0.2504987120628357]], [[-0.3470470607280731]], [[0.3662378787994385]], [[-0.3474118113517761]], [[0.4261741042137146]], [[-0.6116722822189331]], [[-0.4157499074935913]], [[0.37157225608825684]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_59e41e5c9c8c526fffce7e65b3164c76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b79dd7060ce3ed8be9a67757b148a4d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.4468441307544708]], [[0.4903221130371094]], [[0.5349292755126953]], [[0.5423062443733215]], [[0.4022231101989746]], [[0.608171820640564]], [[0.5130035877227783]], [[0.4183555543422699]], [[0.5500997304916382]], [[0.43059056997299194]], [[0.5732475519180298]], [[0.43051764369010925]], [[0.5852348208427429]], [[0.37766554951667786]], [[0.4168500304222107]], [[0.5743144750595093]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class PrimitiveOp_e5374b3b704cfe1b3517932299c864d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0d4b34b5afd8577b7493a42c7f910568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5374b3b704cfe1b3517932299c864d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_ea4814968860531810b95815f2dd8732(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe45d0674cc85e7691adafe359a25cdb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7206a58f51f8e056f60a538c9063d14d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5e00ddb5250321479259e986cdd2528
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_41dc3db919b2e51364e39a3419834472(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a3ff77e2c3b2ffef3d39309f2e443eb
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e0242c1fa2029637fa51c2ec6c7c9569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7996d0f16b68a00175ac2ee58b32986a
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_821bac72dd6ac240e9f77dfc8bb4a634(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be4d5f661cb938009ef15686ba9dbd5f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_0e3223a44d6b149bac782bfc5596abb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_743da1a40e2d519d599bc44cb5426535
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.10938727855682373], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e0242c1fa2029637fa51c2ec6c7c9569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7996d0f16b68a00175ac2ee58b32986a
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_d25b76a7b29fa47dc7b95345c216a542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d2afc835e45d500c9b87ad1ecc06d22
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_828d115ade718aa4fcde66272322f109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d8d55e9b37336a6ae0258500ffa4c7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_58b22091d74839b1bfa470842e4e5aa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d36208301922b0d255a3a1653ba98568
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 18, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d2c918f076d7d4f82ade29a3686b180d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cc472645e01aca6781df184b92f0cfd
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5e48406f87d8dfe173215c62b3b38d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a280b542eb2666a0f42ad377c7d05bb
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5a97e3c13d003fbdf95d3791703211c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8198545b41671385e47b0db8d12c7556
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d45385c53786eecf5b1eda25f491b5fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7fb8394b9cfd76a24b18ef5b1d9966e
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_c57a9000b2c4391ac65a911e66def667(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef01a48e23abc3212b0acf8572ff9e26
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 15, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7004194903d8cf9f1a508a4c35911118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f4d4f7e433464980971678b4bc8860d
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_42b764715d2d3475c83e764c32d049b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79fab2ddab9377dd34e70e31aa78037e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_e7e5682de475dd74f93ca2caad0b65d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5cc0de66dde8037696c9f8a4333633d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 23, 41], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_42b764715d2d3475c83e764c32d049b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79fab2ddab9377dd34e70e31aa78037e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_e3a4f35aae5c0bfe3d5772b288571efe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_126a8025592276f3c231b9564272106f
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 46, 82], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_42b764715d2d3475c83e764c32d049b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79fab2ddab9377dd34e70e31aa78037e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_d47f46cdea8bc4bbb48ea662dd0a5871(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc6aac554478e161f1adc661708a0055
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 92, 164], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_42b764715d2d3475c83e764c32d049b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79fab2ddab9377dd34e70e31aa78037e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_943d991a76559a1ea04ae54266231f65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7179123bbda5856709d935e6dd10d96
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 328], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8397b31463074719157687cebc0dd364(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 24, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_79cb9dda064c6ae9b481c3855f7771c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8397b31463074719157687cebc0dd364
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-4.620937347412109]], [[3.8856356143951416]], [[9.729605674743652]], [[-7.255503177642822]], [[-9.310602188110352]], [[-2.4900877475738525]], [[1.9214890003204346]], [[-7.671777248382568]], [[-6.130800724029541]], [[12.568984031677246]], [[-6.526096820831299]], [[3.945925712585449]], [[-2.178328037261963]], [[1.053225040435791]], [[7.4328389167785645]], [[-7.750824451446533]], [[-0.41833260655403137]], [[2.4057672023773193]], [[-7.5754852294921875]], [[-4.216159820556641]], [[4.076785087585449]], [[-1.4666365385055542]], [[-5.553734302520752]], [[5.240279197692871]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_6b99966b91ad073fc5eed2399f172a86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4b73c091f5ae1b12cd82ee0aff7fda4
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 23, 41], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.0]], [[1.0]], [[1.0]], [[0.0]], [[0.0]], [[0.001982450485229492]], [[0.8842978477478027]], [[0.0]], [[0.0]], [[1.0]], [[0.0]], [[1.0]], [[0.06433439254760742]], [[0.7106450200080872]], [[1.0]], [[0.0]], [[0.41633346676826477]], [[0.9811534881591797]], [[0.0]], [[0.0]], [[1.0]], [[0.20667269825935364]], [[0.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_742fdfa3fa85b751aeff14490e1ff187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8397b31463074719157687cebc0dd364
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-18.66924285888672]], [[15.249897956848145]], [[-20.509868621826172]], [[-18.034530639648438]], [[-15.790414810180664]], [[1.9835697412490845]], [[2.2350118160247803]], [[8.050966262817383]], [[-0.6699670553207397]], [[17.85790252685547]], [[0.8371495008468628]], [[3.0087337493896484]], [[-3.0142767429351807]], [[7.576848030090332]], [[16.059762954711914]], [[-29.40458869934082]], [[-29.24024772644043]], [[-3.7763969898223877]], [[4.680673599243164]], [[-2.011524200439453]], [[21.911062240600586]], [[-12.582876205444336]], [[20.6669921875]], [[20.4222354888916]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_1304944a8e853490a71d444672efb349(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15f7a64633c9f07e2e70f08658436418
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 46, 82], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.0]], [[1.0]], [[0.0]], [[0.0]], [[0.0]], [[0.8967139720916748]], [[0.9470024108886719]], [[1.0]], [[0.3660065829753876]], [[1.0]], [[0.6674299240112305]], [[1.0]], [[0.0]], [[1.0]], [[1.0]], [[0.0]], [[0.0]], [[0.0]], [[1.0]], [[0.09769514203071594]], [[1.0]], [[0.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_2c97c5fdad00424d2ad23c3b1e4ed96e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8397b31463074719157687cebc0dd364
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-1.6105514764785767]], [[-11.617898941040039]], [[-15.899986267089844]], [[12.991196632385254]], [[-4.896671295166016]], [[-1.1534380912780762]], [[47.22460174560547]], [[-33.95370864868164]], [[12.898575782775879]], [[13.548272132873535]], [[-7.006844997406006]], [[-5.018439292907715]], [[14.465497016906738]], [[15.362183570861816]], [[31.865205764770508]], [[6.643660545349121]], [[22.678098678588867]], [[1.4045765399932861]], [[-5.209331512451172]], [[-27.524900436401367]], [[14.777848243713379]], [[-11.47730827331543]], [[-0.9316047430038452]], [[-7.450672149658203]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_ba1fbe7aa3535dff1bee3d49116a5642(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0191fe6a1122ab6596c791553e81af5
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 92, 164], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.17788970470428467]], [[0.0]], [[0.0]], [[1.0]], [[0.0]], [[0.26931238174438477]], [[1.0]], [[0.0]], [[1.0]], [[1.0]], [[0.0]], [[0.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[0.7809153199195862]], [[0.0]], [[0.0]], [[1.0]], [[0.0]], [[0.313679039478302]], [[0.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_ed9f99da8a8b1003e0f0026baa2d06c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8397b31463074719157687cebc0dd364
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-6.0602922439575195]], [[-17.824617385864258]], [[13.900662422180176]], [[-15.259296417236328]], [[6.791174411773682]], [[8.69936752319336]], [[-1.701535940170288]], [[4.16383695602417]], [[-4.911139965057373]], [[4.1500372886657715]], [[5.514294624328613]], [[-1.6808996200561523]], [[-22.533723831176758]], [[10.759888648986816]], [[-3.4057741165161133]], [[-5.675028324127197]], [[-3.314412832260132]], [[-9.971339225769043]], [[12.485906600952148]], [[-9.071209907531738]], [[-7.284687519073486]], [[3.8054251670837402]], [[2.681209087371826]], [[-5.196110725402832]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_73ab2460f1c143c3fa28e8e692c0993a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1abddc10690c7fb3fedd5221fef428ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 184, 328], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.0]], [[0.0]], [[1.0]], [[0.0]], [[1.0]], [[1.0]], [[0.15969279408454895]], [[1.0]], [[0.0]], [[1.0]], [[1.0]], [[0.1638200581073761]], [[0.0]], [[1.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[1.0]], [[0.0]], [[0.0]], [[1.0]], [[1.0]], [[0.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


class TestPrimitiveOp_1bf69e80b35e651422d11175d81880b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([1.1723005771636963], dtype='float32').reshape([1]),
            paddle.to_tensor([0.30833929777145386], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_308077f04746f4c10afd46f849f3af97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7731756567955017], dtype='float32').reshape([1]),
            paddle.to_tensor([-0.4740374982357025], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_13b66a0cd9429694bf23faf779aa77dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([1.2857191562652588], dtype='float32').reshape([1]),
            paddle.to_tensor([0.21764665842056274], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9695b34546fe3be9c77f7b9ea7016f83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([0.6966312527656555], dtype='float32').reshape([1]),
            paddle.to_tensor([0.0015254020690917969], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6614933aa5c88296ffe4cb84de0c8eef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([1.2820426225662231], dtype='float32').reshape([1]),
            paddle.to_tensor([-0.36528217792510986], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c21083c97e59f2381c4147f560924693(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([1.356614589691162], dtype='float32').reshape([1]),
            paddle.to_tensor([0.2626228332519531], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_31f698be545d8c695570ccc9d06ced41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([1.2716164588928223], dtype='float32').reshape([1]),
            paddle.to_tensor([0.06226104497909546], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_a33745bffe6eb98c16d5009a28dd456a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0511736869812012], dtype='float32').reshape([1]),
            paddle.to_tensor([-0.08995750546455383], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_39ba6b6a760b1ce4f052c4beccbe3a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6fff49d58378f993e64f73bb4ee0170
    def get_inputs(self):
        return [
            paddle.to_tensor([1.1593267917633057], dtype='float32').reshape([1]),
            paddle.to_tensor([-0.02210572361946106], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_80f305e057021fb95b8a1e9428af6b80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d17b3eda186f14a705c5a4930d1297b7
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7b83018ec9efa707eb1ea4398f1d9231(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b26322edd22e347c23b13c8c14f672b
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 9216, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4bf47e931f3ea79362691880a6b8f823(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4be8869c253a0f11a062831a46528ee0
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4751e898c29391a9ddb80261f8cbd24f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06e3aeaa24fef692fbebc0f3db6bdcf8
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d45385c53786eecf5b1eda25f491b5fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7fb8394b9cfd76a24b18ef5b1d9966e
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_68d6f9e2c0039f27313cd0c9056cd5ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7f241038e30db9e6912d0782751485c
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_24811e99831e772ab5517a756453c85f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a26dfc3e1ee7aecbc2a9fe0ad4767ca7
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 100, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.37274858355522156], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_eab687913e0e3d772d31dbf05424e461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0f893a89be09dd2f9000d61202b6d7f
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e19dee740eca47ae59647d3ffe4a8d0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12df68d89b8813493227a5d5192e489b
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10e194c8bac5598853dea89b168480b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05eb97b286a792a80f7726c6363886ec
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f3d932ca3fab708661f9d8a4131a18f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1571de3b7504056e385f67f78d5a62f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_a7b69445266fa5c4849a06619cb6371b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_786d583c57b98a30e25d9f6524f2c827
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_36363d7e097a47568edd8771d4a9d17a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2481e105c4881796aef69e0dd25a9d25
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_cbb8a9c2e22e354a365d4fe1d35f0a61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b73eb6d43f9b406cfe532f993fde9f7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a7352c87413d91e5b8b998f2df3273ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15bcbb6414ff84a0d699fdaafd45d2b5
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_da31bf5d6a04726a891465e7ec6ae534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39231879b907c4e74184405b5f147722
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 160, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_825bb9e1509de05282561681030b3295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9304bb250d6d76ef639efc1e1e0c1339
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([256], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_f5437a2ac81800e5c9c63798e24dafa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32089dd975aa5ae09d0e77a4bb9afc16
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.3945217728614807], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f5437a2ac81800e5c9c63798e24dafa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32089dd975aa5ae09d0e77a4bb9afc16
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.3945217728614807], dtype='float32').reshape([1]),
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


class TestPrimitiveOp_dc5769a5184a761fbf3fcaee89955de7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f77c72d9dd9c5b058b46aab99ecf9960
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_e8c87189639dd2888882901c43bc8413(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96bdec893315e84077b46ee1d2a7eea7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.11497446894645691], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_990786bb38024e86bf09b9526a4019cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23efb7705a00d013dde414a35a50893b
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_e42f2c0bf2f03b4cbf24eca3cfd8b2e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_781ca45ad191072a9fda64884f7b50ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 104, 104], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_799869d910bd86410c260f157e508c92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0dda6823b552830379006659d462860d
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([16384, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b31fe694663a63c77eaf9110c02771f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79fab2ddab9377dd34e70e31aa78037e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_694cd6f515e023d37337eb84c13a43a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91c445d21139e34473e8f53ae00a1e56
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 104, 104], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7d136372223a8a15f4c6a86a1d669690(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51c40fa1586b97a36bcde0f702df97b1
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_def03e62a4814d783b71ad29c5d28820(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7863b2beff958c4ac56a444880cee21e
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_331180ea98e0a80a458572ff2acd765a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6703f810822211bca67b8219701506ea
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_81d127e66eeaf3a06d2a06fdf9e5c504(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dd8f5115e9b9b191e1a05315edd9c17
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_3b3c0e42a2c914983bb65bcf85ccf318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8180e4b8ab9a80c3e815e033d0fdca0e
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e1e4cf7960f8860e4f589cb4480e9022(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1272b57e0e52d16007171347313276a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e1e4cf7960f8860e4f589cb4480e9022(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1272b57e0e52d16007171347313276a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e1e4cf7960f8860e4f589cb4480e9022(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1272b57e0e52d16007171347313276a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e1e4cf7960f8860e4f589cb4480e9022(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1272b57e0e52d16007171347313276a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4df27415c72a07f3045dd0694737537a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c19d801d8a9adac0c877ec38bfa79311
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_6e20ce5303fe2ce7e8f8a80de7140b1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a68a05971af9c854fa27a115447b4d5c
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_dda657fc3ba4d4835495f3f2bca326d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df7011d3bb16432e13116eb1e12c4c50
    def get_inputs(self):
        return [
            paddle.uniform([70, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([70, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dda657fc3ba4d4835495f3f2bca326d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df7011d3bb16432e13116eb1e12c4c50
    def get_inputs(self):
        return [
            paddle.uniform([70, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([70, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c6631963efe711e78cba6954881526b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[70, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00872609553be040b6e25687f61faf36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6631963efe711e78cba6954881526b4
    def get_inputs(self):
        return [
            paddle.to_tensor(-1.0, dtype='float32').reshape([]),
            paddle.uniform([70, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dda657fc3ba4d4835495f3f2bca326d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df7011d3bb16432e13116eb1e12c4c50
    def get_inputs(self):
        return [
            paddle.uniform([70, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([70, 81], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_e16f5ac73ed430d69371a734803324b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_210051582262298800fe8fc79767c3f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.10758101940155029], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_a94bae55e178ce795ec41a2f815b07ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 144, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 144, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_91e89c3de7295306809e718aa0403a5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a94bae55e178ce795ec41a2f815b07ea
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 144, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_603c29d0be00097bb05c2104e2cfa2ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 144, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[6, 144, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_10ebbd2ece0100ec87618d93d27be72c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_603c29d0be00097bb05c2104e2cfa2ce
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([6, 144, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_872dd00544fafa63faaef4d038f9bdfd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6, 144, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_369a6c7cf161531ee9f8f2066dba48e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_872dd00544fafa63faaef4d038f9bdfd
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b31fe694663a63c77eaf9110c02771f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79fab2ddab9377dd34e70e31aa78037e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_1926ef0f33b43249d226e8981a4f2f1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43a769bc28dbce6a70906be4d6c49877
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 184, 184], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f3d932ca3fab708661f9d8a4131a18f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1571de3b7504056e385f67f78d5a62f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_ea7520c24bc9727f0ff9ba4dd3bc034b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8c4d4815ed87c008011679ed4482b0f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 34, 34], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d8d474a6109712b807070ac7fc376a9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b20b0336e12a00d9214d414a53a93361
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([15200], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_42b764715d2d3475c83e764c32d049b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79fab2ddab9377dd34e70e31aa78037e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_b823e8dc15ecd9aa6103f5b85b0ffd8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5f2ac5a2cdb4bb1c7e477fd6827e878
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_784334a72027c62b543aefc20fba81e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c713bdacca1358dc085fa03dedacad1d
    def get_inputs(self):
        return [
            paddle.uniform([22, 480, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_9482364980ac346090ed507c607fcb70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7e98117ef3be0fa02b989698f62029b
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fdb885db47ca7519ffce04f41009849b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6ca159074f8fe14958b4df4adf001e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_c8b04f79cc8b246ed2c016cd6c00a856(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c6280990710f6cdd77bc4f0ea46c442
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 22, 22], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_117626554b72b339df0699319c7e1acd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2aa2f45786454b6fd609b9ee6d988bf
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([150], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7db91a8ff2795a6877e737980c6964af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626ae5005bb74d0d19e833cd0b4aec1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7db91a8ff2795a6877e737980c6964af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626ae5005bb74d0d19e833cd0b4aec1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7db91a8ff2795a6877e737980c6964af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626ae5005bb74d0d19e833cd0b4aec1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7db91a8ff2795a6877e737980c6964af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626ae5005bb74d0d19e833cd0b4aec1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7db91a8ff2795a6877e737980c6964af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_626ae5005bb74d0d19e833cd0b4aec1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_71abb97d40aad81feaa7ac7585e993ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c97cea5ab0e5517d2625e1cac8042a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_4e5f41604e14ffb3ad7c7692bf98ba65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd1476e44f472b1b8077de762360c50b
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e48853b152277be664fdf715e22fce81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61974deeb2d3a4aa863bbf228e52e4dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_38836ceaf4aea43de992a0d53301e4d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb69369f90a7508e0efc5e3cddd2dbd1
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2972411df7bf6b5462176aadd502abc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88e51f2bae392ed7d4c6b58929ef6a83
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_cd428ed913b1708341ee7a0b2f4568a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac1a626eb1cf5bdff34976e8f96366d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f299bf8db4d3ef58ab6b97fe27a4d977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d6887c242d5b454eb0fc93d223a17a
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8a0d50245c6c7f2c4151dd2899334de9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b69696aecf71a1144fbc6bd6e027ab86
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cf5545733de5d34e5b11306386cdc997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cff4421ec94fec32773214adaf02baab
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c30355eee7f2640258da76929d2f3298(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68e96d9ec0d1c437603475b6006b867d
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c06901c15c1248c6a176e101b19fc431(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf5e69266b9c161f750432b2cda26b49
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_efc6a9c94b8ce79c9be986013405e141(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2dad5f95ffd3ba554f6eeaa58d437952
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f77178fcfcb5907dac6ce17c63cdab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6f4f2695ad16c7c77e69861404b436
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 1, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b31fe694663a63c77eaf9110c02771f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79fab2ddab9377dd34e70e31aa78037e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_ec47b8fc8b8585c4fac023f6c26ecb30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_155232ba38a5736633ca96c889848951
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 76, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d10ea13b4d4128d62dfb4a76605a0060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b4af2e724b4b2010f490551ae44b0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d10ea13b4d4128d62dfb4a76605a0060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b4af2e724b4b2010f490551ae44b0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d10ea13b4d4128d62dfb4a76605a0060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b4af2e724b4b2010f490551ae44b0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d10ea13b4d4128d62dfb4a76605a0060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b4af2e724b4b2010f490551ae44b0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d10ea13b4d4128d62dfb4a76605a0060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b4af2e724b4b2010f490551ae44b0a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_7c24b4296dcaa97919a8daea5d99a9a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7606825c99fce1b673f6c40ad63754af
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7d136372223a8a15f4c6a86a1d669690(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51c40fa1586b97a36bcde0f702df97b1
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_def03e62a4814d783b71ad29c5d28820(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7863b2beff958c4ac56a444880cee21e
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_331180ea98e0a80a458572ff2acd765a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6703f810822211bca67b8219701506ea
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1551fe623e3fee30f40cae16e8d39ff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf6ae6ee44a92275d56a63ee17bb8b8e
    def get_inputs(self):
        return [
            paddle.to_tensor([4224], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_402e80263b0a070212efd91c1e1e8cc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1c061067aeda45f2a5463adabe5015b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1c061067aeda45f2a5463adabe5015b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1c061067aeda45f2a5463adabe5015b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1c061067aeda45f2a5463adabe5015b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ed1d25883866393edc4f976ae5f9364b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([4224, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ed1d25883866393edc4f976ae5f9364b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([4224, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1c061067aeda45f2a5463adabe5015b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4224, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8407e2d8d2f2d788039899ada63862c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba5cc92704ec940eeadc35b0cfa3e2ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_8cb67bef65c56fe73ef073bad06cb8f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_429f77e7f7302cf79fc882d07e53131d
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d79e6833d3f91e08207755f2dfeb43d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5328717a8cee22cfcdd245b743fb930d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_a46d0f7d76c2540fcfce8bb7fb121384(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_002eb7d3088a9f42b9dd7b94a65a4e24
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 256, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b31fe694663a63c77eaf9110c02771f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79fab2ddab9377dd34e70e31aa78037e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_49c044b0bc7b1c30199485488c0bba29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f86f981c78f225e318788a1f9d153c9d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 16, 16], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8fc57001711b806a6526ca853ea32bd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05b9bb8eae4016c3fd8b4548ed85dab6
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_790b39e3e89f7e03563138cd27d97ce8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c45914fe8b40a284028e09cbdb673ee
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be67299aa43f002268392a6c49d469ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1677f2290efe9a742560ef7d7dfb3ba
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_19f33c7b3c45ba488a6506ae11a7c12a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83c3cdb6e829f210bfd695bbc50ad272
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_19f33c7b3c45ba488a6506ae11a7c12a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83c3cdb6e829f210bfd695bbc50ad272
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_19f33c7b3c45ba488a6506ae11a7c12a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83c3cdb6e829f210bfd695bbc50ad272
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_19f33c7b3c45ba488a6506ae11a7c12a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83c3cdb6e829f210bfd695bbc50ad272
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_19f33c7b3c45ba488a6506ae11a7c12a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83c3cdb6e829f210bfd695bbc50ad272
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_322aa25f58dfdf3f0c0d9727256945ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f75c0dd4434232b4fd14b27654a58105
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_0aabcd17915c7ffbfeaa523049a06fe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a42a2723e1e5d7cfc7f59b0c8067d4b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 9, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc219cf526e458de33df21ff9de38ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382b20b4b9f61800ddbe43e1cf3aa712
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_f0c5565ea53aa0bc8d459c37819962ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f80d0ab208ce3eed5377f86b6cbe16fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 64, 64], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e0242c1fa2029637fa51c2ec6c7c9569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7996d0f16b68a00175ac2ee58b32986a
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_5fd7aaa095395fcc0adfc0ed2d74e624(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9751a1a0e3b426334234fe71bf0d4215
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_990786bb38024e86bf09b9526a4019cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23efb7705a00d013dde414a35a50893b
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_f4c5a6ad2fc5962e02a589d44d13d0bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4aa673b6f0d385e88c4482bb417d5b5f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_fbf792b59bb11a280bc95510f3520a43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67dc2be8bf074d228354b5ae558bbe37
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fbf792b59bb11a280bc95510f3520a43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67dc2be8bf074d228354b5ae558bbe37
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fbf792b59bb11a280bc95510f3520a43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67dc2be8bf074d228354b5ae558bbe37
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f49d945a0004c59f99b0605b59d6c1ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 64, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8db43949f57277776a85aed944c38205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f49d945a0004c59f99b0605b59d6c1ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_b6b3f44e982e172f4149a6b9bc22254b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7292c006e54e0195a64c3cccb3a02fb4
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4096, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2972411df7bf6b5462176aadd502abc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88e51f2bae392ed7d4c6b58929ef6a83
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_a9ed440a3b7c23ae867e5187ce97276e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6c0aca5ce2e13946369cfb6c387e262
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d2c918f076d7d4f82ade29a3686b180d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cc472645e01aca6781df184b92f0cfd
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5e48406f87d8dfe173215c62b3b38d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a280b542eb2666a0f42ad377c7d05bb
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 196, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5a97e3c13d003fbdf95d3791703211c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8198545b41671385e47b0db8d12c7556
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_ca8f8d5ba81f58959e89de56473676cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66795651b6916f9d58c606f9f42ec042
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([40], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_556d2b86cf964355374dcba5abc72a12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8736aa72e7a4f74c9b56662baf01338a
    def get_inputs(self):
        return [
            paddle.uniform([1, 92, 40, 40], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 92, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8407e2d8d2f2d788039899ada63862c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba5cc92704ec940eeadc35b0cfa3e2ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_131403c738a2d2a190722eecf57248eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9be192a628edefe956fdca020ebee958
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 30, 30], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_91c81fe6df3862f592f7a40d56451293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61974deeb2d3a4aa863bbf228e52e4dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_d7199bb25eb0b7f73c31364686f76dfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2080653d0e0c3645fa313a40c82a396b
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0d7f364a9170607d8d531cccd2ff517d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 49, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_71a0271c32f1868cf5298fdda64c0942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d7f364a9170607d8d531cccd2ff517d
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_11ecfe7fc9ed00cff6c23847058aacc0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[11, 49, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab4afd9813b2c5ed5d5adf3024bbb947(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11ecfe7fc9ed00cff6c23847058aacc0
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_0757769494dfa474756245efcd77757c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[11, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f45886a042692a7a71c7efc19cadf103(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0757769494dfa474756245efcd77757c
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_8919622cbf8dde265b22178a6c697e44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba81e29fe6eac68f9bfbc2813a1c4bb2
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 36, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_db5eeaa89595dcd49f685e02115b75bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c3d536f6cc02ffd67c2d6197f0530bc
    def get_inputs(self):
        return [
            paddle.uniform([3800, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_db5eeaa89595dcd49f685e02115b75bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c3d536f6cc02ffd67c2d6197f0530bc
    def get_inputs(self):
        return [
            paddle.uniform([3800, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f72a83c1a9999150e6d00eceff7e7339(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[3800, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ee98e2a14892730a191c3110d3c69fed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f72a83c1a9999150e6d00eceff7e7339
    def get_inputs(self):
        return [
            paddle.to_tensor(-1.0, dtype='float32').reshape([]),
            paddle.uniform([3800, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_db5eeaa89595dcd49f685e02115b75bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c3d536f6cc02ffd67c2d6197f0530bc
    def get_inputs(self):
        return [
            paddle.uniform([3800, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_322aa25f58dfdf3f0c0d9727256945ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f75c0dd4434232b4fd14b27654a58105
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_6a66c70fd180b4cf470bfb646ef331d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f44a67f1dfb1f50ea23906280e5ff14
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 13, 13], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e0242c1fa2029637fa51c2ec6c7c9569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7996d0f16b68a00175ac2ee58b32986a
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_54c8098101ce23555018b5713abb899f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_502a7eb46deb36385e27cdfcc2c1fa37
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 92, 92], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f299bf8db4d3ef58ab6b97fe27a4d977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d6887c242d5b454eb0fc93d223a17a
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8a0d50245c6c7f2c4151dd2899334de9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b69696aecf71a1144fbc6bd6e027ab86
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 320, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cf5545733de5d34e5b11306386cdc997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cff4421ec94fec32773214adaf02baab
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a0ec15511522c7f16a8e57788ac7e978(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2112e1a50a0c54fdd01e3273ef3010a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0ec15511522c7f16a8e57788ac7e978
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_867099412d2f51930dc684e80860976d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0616a5ceabe962aa3b2a8edd4642fa70
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1b293d4078c1ae5006e856954c541dec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 144, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 144, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59ffbf64d6f8dc0b828187f6515c7c77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b293d4078c1ae5006e856954c541dec
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 144, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d3f64dd4239b5fbf0316b049f8ec0d66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 144, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 144, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_edc16c14f5e84744a55b4e2e28d231ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3f64dd4239b5fbf0316b049f8ec0d66
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 144, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_25a79b30cec2c1bf5d00846a5681c677(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 144, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c06a2c294bae8543ebb535763cc9326(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25a79b30cec2c1bf5d00846a5681c677
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1138f2f484ffda0d688181e6ab0d12e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3202affc3aba5367ae9a98124a1547b7
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_af614c28e033c231e0dea75fe9686109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22aec529d4321b5699d224d6ffb96e74
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 25, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([0.46657663583755493], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9e08a978daece187dd1935def49f8015(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a68ea25b83d6679299370ee7f9ec504
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_c72c7b47e8679bfb0746ae950f77252c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99762ac386920a1faa3d3da1d7e6c1df
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 26, 26], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e0242c1fa2029637fa51c2ec6c7c9569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7996d0f16b68a00175ac2ee58b32986a
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_6c940fb2ec4abb66a035b781ced62188(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c7bdc80da0b2006252a28c70adb41a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_7c7b0e24f4c52bd648cf3021326c8b3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27041bd2d04e95bf6250eed180c2bd1c
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3800], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a41446fcfe7b0961bb461415732f6456(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf6ae6ee44a92275d56a63ee17bb8b8e
    def get_inputs(self):
        return [
            paddle.to_tensor([4657], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_9229e40e712701c1a97430ee5cfeb2aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_993e5465932dc7fe503a6d8c702522f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_993e5465932dc7fe503a6d8c702522f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_993e5465932dc7fe503a6d8c702522f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_993e5465932dc7fe503a6d8c702522f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_94df28d7ac3ccf435f4db66e9a469b40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([4657, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_94df28d7ac3ccf435f4db66e9a469b40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([4657, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_993e5465932dc7fe503a6d8c702522f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4657, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1e50b69c3dfb0d41bd003f038aff6df8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a2899a2d1393f5bd93c18a1b5691484c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e50b69c3dfb0d41bd003f038aff6df8
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5c11c49db9466085676dcd698085c0c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d408b3585b9d2a7f04f90bc9650d733c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c11c49db9466085676dcd698085c0c6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a7ba08b32f5d51d68f5046b8d44eda32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_939bbffd213fca9cea1be55d5fbc1c0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7ba08b32f5d51d68f5046b8d44eda32
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_08518e09f71108972795f63c3845623d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57d13157208d60ca4bba4a608f1cd7e1
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_caca49919a414e392b247247391ffbd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74df853cb3e234fdc02f2538a0f4a188
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 784, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ff172bb99fda2b29a788c66e30fd581e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_566f215bc9b6fe9cbbcece1bb178a5b8
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_71a0271c32f1868cf5298fdda64c0942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d7f364a9170607d8d531cccd2ff517d
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ab4afd9813b2c5ed5d5adf3024bbb947(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11ecfe7fc9ed00cff6c23847058aacc0
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([11, 49, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f45886a042692a7a71c7efc19cadf103(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0757769494dfa474756245efcd77757c
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_bbf15e0683684e8efe47860a80ae25a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c7b980800ab890863c28f9a24ff437a
    def get_inputs(self):
        return [
            paddle.uniform([171, 36, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 36, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3356594debffdf46265e4bfc50b7421a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf6ae6ee44a92275d56a63ee17bb8b8e
    def get_inputs(self):
        return [
            paddle.to_tensor([3770], dtype='int32').reshape([1]),
            paddle.to_tensor([4], dtype='int32').reshape([1]),
        ]


class TestPrimitiveOp_e7dd58045e34d5f24b91e79a15aad753(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_476e6620e1438088f38ad912418c2905
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c44337679c0bde0428182ceb31168fc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c44337679c0bde0428182ceb31168fc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c44337679c0bde0428182ceb31168fc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c44337679c0bde0428182ceb31168fc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10a9ef1cf35b0f0ac764b7fc436b0617(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([3770, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10a9ef1cf35b0f0ac764b7fc436b0617(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2c054a0fdfb6cb0cd90fc09624fb2db
    def get_inputs(self):
        return [
            paddle.uniform([3770, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c44337679c0bde0428182ceb31168fc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efb9086f821a20072586613485e9cf86
    def get_inputs(self):
        return [
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([3770, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_64dd70942b2bc701cfbe4938d1fe843d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3f803a756869e501116ea42698960ea
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1855833d1a5d0b933bfb611165712f81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_057715d92dae638dcf6709c1155905fc
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 100, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_57bee0f8676a7a7f95e587c0f67f4309(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8bcd5775fc0db4d50eb321309514e9e
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([128], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_888578a955b3d87190aa48f683730ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fcbc8e940e1477629be8048e2b3c30b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1248, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3086425fc951090c34f0bf80c1f10896(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c514e2a5c05418dc0e1b53b10e40f6b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_62b726cf4ffd45b7200a5d090ab30e75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc3ba840de278f277d1bfb295374b94e
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_33f2352a28b5bfa521d3eff0ad30a123(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06afa0ac8681719902f3e86d0feb1b9c
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([[[[0.4779541790485382]], [[0.6030649542808533]], [[0.44898539781570435]], [[0.5123769640922546]], [[0.4052342474460602]], [[0.5249818563461304]], [[0.38641709089279175]], [[0.5528154373168945]], [[0.45246651768684387]], [[0.5779471397399902]], [[0.48136067390441895]], [[0.5380899310112]], [[0.46596118807792664]], [[0.5732711553573608]], [[0.5935705304145813]], [[0.3933275640010834]], [[0.43395718932151794]], [[0.45972102880477905]], [[0.5645336508750916]], [[0.5756213068962097]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_e8d878ebb439c20b2ae99967473a9d37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08a35fac70f021394b01b6711933f04f
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_b11978ef05625070cd771ec1592a8d62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_396325b24b038419a11b6092c7a134be
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 64, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1b4b78f6dc93dd6c888b481651b7c68c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1174, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9cdf4ff9ce03bf076f44d8287619474d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b4b78f6dc93dd6c888b481651b7c68c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1174, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6efecebe1a697d6569c87249a3c106a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1174, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e8faabe16f9e31b3e514c664a6e9af6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6efecebe1a697d6569c87249a3c106a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 1174, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_54402d46624977bc2ff935f5db3f9b18(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1174, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2271dacb840c4cc468705fdb176fe3a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54402d46624977bc2ff935f5db3f9b18
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e27b4de0cc9dcbf2abd876033a414f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_094c190833ee8d62c1961fdea27f1fae
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e27b4de0cc9dcbf2abd876033a414f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_094c190833ee8d62c1961fdea27f1fae
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e27b4de0cc9dcbf2abd876033a414f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_094c190833ee8d62c1961fdea27f1fae
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e27b4de0cc9dcbf2abd876033a414f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_094c190833ee8d62c1961fdea27f1fae
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e27b4de0cc9dcbf2abd876033a414f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_094c190833ee8d62c1961fdea27f1fae
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_4eba9e297eb021789f838eec4f83de36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f48b73f59d09160632ece643fcc9242
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_1e5da005497742a9731ef5d4ecc2e16b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d15fe55ef3097276ca57da60c4d5cba8
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_18f3ec25c187d86de9b89c8ac02b696a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3630186f00bb00d90ccb305173718830
    def get_inputs(self):
        return [
            paddle.uniform([145, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([145, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e0242c1fa2029637fa51c2ec6c7c9569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7996d0f16b68a00175ac2ee58b32986a
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_10c4749d295c0083152e623dcc448dae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1302bca5f85a2e2cee7fc197723ff908
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_72eca9e36d606e8a0a9177147ee61616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9914dc33ff65efd9e6c755c3d3b201f5
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e0242c1fa2029637fa51c2ec6c7c9569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7996d0f16b68a00175ac2ee58b32986a
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_5dacc36f67b9dbb8904ce3d13e775a14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4282a9406db450d904526d6461a1776
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c33240ccb715a70d663d4e013486963a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce361428574103724a95686f3ca8862b
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_87de9916d58a93a2e8309db1bcd73865(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d1d45e84957895b7ac2b031636b2994
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_386839899c9f9a689154df9cfaa7e532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de1e7f314ab59f7cc77bd918188edb3
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fc219cf526e458de33df21ff9de38ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_382b20b4b9f61800ddbe43e1cf3aa712
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_a86acf496c3da90f50b9351d99d71ae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aab31f782e3fa23dc3787266af269c40
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d725ad75482a80a14e59b3b3c00fc205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f75c0dd4434232b4fd14b27654a58105
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_2d5de2cfbd99a8bcfb1fc833a23231e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83ab3565ebf3df8809d393520af0ed08
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_990786bb38024e86bf09b9526a4019cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23efb7705a00d013dde414a35a50893b
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_e8d37faa42bcd4ffeb9c29932d7e9e2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a86912fba1468a4cc752c058a2e18296
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 152, 152], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_d5892c3d0b40fd3d3e057e474f88e0ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 576, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 576, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_086da90e0a5df71e19792c28308b455d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5892c3d0b40fd3d3e057e474f88e0ab
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 576, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_7d8f88139d919695b19a2dcc46da13bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 576, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[4, 576, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c9402e2e12950b9b055325f9a7ae6ec8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d8f88139d919695b19a2dcc46da13bc
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([4, 576, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a1327f1799f9b78d73ea140e77cc8754(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4, 576, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_abd492da1ff2042494c542d893fb7a37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1327f1799f9b78d73ea140e77cc8754
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 384], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([384], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_258a088123cae70370338285b5d5dc17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8272a0029c17a0981d6dc0bb8f84c77b
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_d560a4b4f3f860df6717e3255128e747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_baeabc11c986938180f5901cbd5e2eb1
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1024, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d725ad75482a80a14e59b3b3c00fc205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f75c0dd4434232b4fd14b27654a58105
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_3f80c33f13528440f791250ddc9c66d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8098960eb115a79955652f745c57d4f2
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 22, 22], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_d2cc393b99300e17e701d60ffa567439(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c12ca22a303ef482473d6959339461cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.11142927408218384], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_5846997ad303bb3f1611c37032658b33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8253ad6111de39b8930be7b782affcb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5846997ad303bb3f1611c37032658b33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8253ad6111de39b8930be7b782affcb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5846997ad303bb3f1611c37032658b33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8253ad6111de39b8930be7b782affcb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5846997ad303bb3f1611c37032658b33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8253ad6111de39b8930be7b782affcb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d45ba9b26809ee86adf3dadcc6e03895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7541fd6af8509f0ce8734aa60e436973
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_c2a9813b92013dee91ea7d7ad385c72e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_515ba5bae4ea344adfb82b718c674e68
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_c1c5d15315ae7f64bddf166e9b2c487c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9977247759bc092dedb9973d521f77e
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 1, 49, 56, 56], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_655fae65f2acc98c547a837422c0b500(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_194b11587c47088f1c2131d4d4ac0ce9
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_77e3e0210b0547fe5f4522460de1a28d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_711d61c2ab9ee86e44eb33a3f879f802
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 60, 60], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_71abb97d40aad81feaa7ac7585e993ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c97cea5ab0e5517d2625e1cac8042a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_bbddaeb33668f1977dfefe476f7288f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3adb20594a6eb3588fd3da022747d609
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 48, 48], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_f36dccbd4db6a64b8d307d43fce8de88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7bb475a7714f3d482ff50c213b4d533
    def get_inputs(self):
        return [
            paddle.uniform([10, 480, 7, 7], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([10, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e5f83a1ca6bd07617247d9db08a03faa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acb75ac7d8898f56f550b9ebe95dc4b7
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b31fe694663a63c77eaf9110c02771f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79fab2ddab9377dd34e70e31aa78037e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_4eba9e297eb021789f838eec4f83de36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f48b73f59d09160632ece643fcc9242
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_4dd93ba9b76a46dc5be72f7ed83d38be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1ad8cfdf8ad799f5c7266cc0e406353
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4dd93ba9b76a46dc5be72f7ed83d38be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1ad8cfdf8ad799f5c7266cc0e406353
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4dd93ba9b76a46dc5be72f7ed83d38be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1ad8cfdf8ad799f5c7266cc0e406353
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_ad0c88185eaacc57e8728775a8205f71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77abb17d1cf32ac968d3cf6df91220de
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.20353367924690247], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_0e88311fe9f97ebaa89675abb18aafe9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 38, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 512, 38, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_75c7bcc6cd4810813f4a731c88d86646(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e88311fe9f97ebaa89675abb18aafe9
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_521046fff837d0571261457a39f8f347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9869cb9d0710d9208427919ef767c7ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c33240ccb715a70d663d4e013486963a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce361428574103724a95686f3ca8862b
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_87de9916d58a93a2e8309db1bcd73865(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d1d45e84957895b7ac2b031636b2994
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([43, 3136, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_386839899c9f9a689154df9cfaa7e532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5de1e7f314ab59f7cc77bd918188edb3
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_80af52dfe292d72107a04668b0cd1343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5328717a8cee22cfcdd245b743fb930d
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_05242b284b6c70801ee622ada37f1d7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd5bc996e3ff67991d0ac104d81f369b
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 32, 32], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_19e5f6888eeb2fff3b87db49c4b09c28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5e21e9893a0ee7929b613cf12ea4de8
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_19e5f6888eeb2fff3b87db49c4b09c28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5e21e9893a0ee7929b613cf12ea4de8
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_19e5f6888eeb2fff3b87db49c4b09c28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5e21e9893a0ee7929b613cf12ea4de8
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_de275957176cabe944712ce774e4dedc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac79f51688468978dfb02ba6eed85997
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.20000000298023224, dtype='float32').reshape([]),
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


class TestPrimitiveOp_6579dc2c7d1d487ed43ed103fa8f737d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9902438db4ef593149b654f30dd9b70b
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 10, 10], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d45385c53786eecf5b1eda25f491b5fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7fb8394b9cfd76a24b18ef5b1d9966e
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_6ff998b71e60911c174ac206ac1a2c66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5483c1196783a3f5fa34f740c0f96af4
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 12, 12], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_4c3fde04ca826e1430cd3419693725a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15d40efaf2971c8ae33fea5f84371db1
    def get_inputs(self):
        return [
            paddle.uniform([247, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([247, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c3fde04ca826e1430cd3419693725a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15d40efaf2971c8ae33fea5f84371db1
    def get_inputs(self):
        return [
            paddle.uniform([247, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([247, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9c33d5711685e9ed640b85f3aa8e34e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[247, 81], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ca0f65abd556f422c33fe75d6bfec752(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c33d5711685e9ed640b85f3aa8e34e4
    def get_inputs(self):
        return [
            paddle.to_tensor(-1.0, dtype='float32').reshape([]),
            paddle.uniform([247, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4c3fde04ca826e1430cd3419693725a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15d40efaf2971c8ae33fea5f84371db1
    def get_inputs(self):
        return [
            paddle.uniform([247, 81], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([247, 81], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0d4b34b5afd8577b7493a42c7f910568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5374b3b704cfe1b3517932299c864d4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_b9348641f6af4ef8e5cfdfe993b11b4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f42814ead87e7fa1a2c611bd00bbd84c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 15], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4fe93ab0db01dd2a2aaec1c7bbe3ddcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c19d801d8a9adac0c877ec38bfa79311
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor(0.16666699945926666, dtype='float32').reshape([]),
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


class TestPrimitiveOp_c024b8e7ffa09dd73f90cfa87be417f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17973d10c572415966e8a183af596682
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 18, 18], dtype='float32', min=-0.5, max=0.5),
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_0b95ca3eb03ca2e9890c0b20aaddef37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f74d0ba23ea63a3e855d3b191829cd80
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=-0.5, max=0.5),
            paddle.to_tensor([-0.015302866697311401], dtype='float32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()