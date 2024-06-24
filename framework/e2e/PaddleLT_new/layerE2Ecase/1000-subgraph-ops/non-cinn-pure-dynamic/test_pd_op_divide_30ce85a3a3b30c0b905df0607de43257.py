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



class PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07bcf08a2d2fcb66983dfec5780923d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d6ebd010a5bb57e1b0bc1cf9a08133b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_31c377cf042b9cb40b1e0a7567853e6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24010474979877472]]], dtype='float32').reshape([1, 1, 1]),
        ]


class PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_49e0d10ad1dbacff76977598eb08e0d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([12096, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_309f51c50c9850cd16dd3894e8a41433(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7847848f8dae9ca754775f32e227052(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_703b9c4a7e8742899c426a973f3f855f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9ac62f10544f515017c078e18a1aab64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_703b9c4a7e8742899c426a973f3f855f
    def get_inputs(self):
        return [
            paddle.to_tensor([1116.469970703125], dtype='float32').reshape([1]),
            paddle.to_tensor(8732.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_5689945fabeb564098bee437186ea5ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.0007042176439426839], [0.002781935501843691], [0.005191003903746605], [1.3299752026796341e-05], [0.00800674594938755], [0.029768511652946472]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_1caf1ca65cc5c70d408b39c8d0b0699b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.00017917151853907853], [0.0015084518818184733], [0.0010314411483705044], [0.0049806316383183], [0.022327452898025513], [0.0007504618261009455]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_792537da402a7d37595e4c8b9225b1c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.017739005386829376], [0.04480043426156044], [0.03680412098765373], [0.10784060508012772], [0.06260087341070175], [0.10046130418777466]]], dtype='float32').reshape([1, 6, 1]),
        ]


class TestPrimitiveOp_cb7948835bad5bd55157e3be8dc9c037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f69ac909d4915371a12db1de0ee05cc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07bcf08a2d2fcb66983dfec5780923d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc712d615bb205a08ac859dc5b80a517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c7c07c53a0fe3540238159515a9fda69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0aff9a3ba9d524f189146a1aa2445d3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(9.130020141601562, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_010b1467acb3db2c29b09a9323fb4383(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(2.597266674041748, dtype='float32').reshape([]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e4e8ccbae22d55caec42f8915a479c02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47c29bffa0f6434e6dbc903b79f34f98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_47c29bffa0f6434e6dbc903b79f34f98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1745, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cdcfbbbd2ca7a5337dd59c44bbb8da51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(77179.875, dtype='float32').reshape([]),
            paddle.to_tensor([0.3140214681625366], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bd31882783d3588bebf61a16b470b7ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(99044.0625, dtype='float32').reshape([]),
            paddle.to_tensor([0.3140214681625366], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_808c3b5a8572ecc7826d0aed30fec03e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(937.7254638671875, dtype='float32').reshape([]),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9871bf1396c8b9db33c6571fcb21ce8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7364055f7814905dc4901270efaeb86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eb9d6efea643bf5df74687e22793215f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([5376, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c80361c71d4987f902260c9caae754f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcc39eb9f4f10134caf76738aca0515f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f347c818d2ad81ca0850ce3d337d649(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.037523772567510605], [0.0], [0.0], [0.007898855023086071], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.08137653023004532], [-0.05491970106959343], [-0.010116374120116234], [-0.031606074422597885], [0.11603888869285583], [-0.03504899889230728], [0.028307661414146423], [0.18799147009849548], [-0.011176660656929016]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_43d0e6904ad1badc24a1dcfda20b4275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.039160557091236115], [0.09260876476764679], [0.049346014857292175], [0.029952984303236008], [0.006903171539306641], [0.15719175338745117], [0.022874511778354645], [0.0], [0.030478760600090027]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.042215973138809204], [0.03768906369805336], [0.03922963887453079], [-0.0016530902357771993], [0.12294206023216248], [0.12214275449514389], [0.05118217319250107], [0.18799147009849548], [0.01930209994316101]], dtype='float32').reshape([9, 1]),
        ]


class PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4beddcad013ca657dd71021e464d762a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.368469774723053], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_54b81f84dd67aea6b80d5f281cf4cb82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54b81f84dd67aea6b80d5f281cf4cb82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5556, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c1f8238fc09ea1d7c7cd97e44a725dfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(54545.875, dtype='float32').reshape([]),
            paddle.to_tensor([0.16813154518604279], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e8a5b9e2dbeaf2f2b7579b41801c7ff1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(3950.26953125, dtype='float32').reshape([]),
            paddle.to_tensor([0.16813154518604279], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_ff10856afc23f7aff18e2b675cc93008(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_879ad54f2a997ba12891ab2e98d4c149(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.0, -0.0, -0.0, -0.0, -0.0, -0.0], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.00635618856176734, 0.0034854901023209095, 0.019583148881793022, -0.03482642397284508, -0.03631962463259697, 0.0001548960426589474], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_754c5a8f07e93e331e49a4c95eb4eb95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([0.011698182672262192, 0.03742148354649544, 0.03183488920331001, 0.0026579787954688072, 0.0013092659646645188, 0.01345645822584629], dtype='float32').reshape([6]),
            paddle.to_tensor([0.005829278379678726, 0.17400214076042175, 0.2847479581832886, 0.12188687920570374, 0.01871480792760849, 0.13883072137832642], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ee252490c50797b911038caa3e595964(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([0.052370764315128326, 0.19320246577262878, 0.1291169971227646, 0.3269874155521393, -0.2654902935028076, 0.19371631741523743], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.12136902660131454, -0.03549548238515854, 0.15166980028152466, -0.10650692880153656, 0.136802077293396, 0.0007996018975973129], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_287882391f367ebfde0139a96408ee75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.017781324684619904, 0.02797788381576538, 0.44838252663612366, -0.032613471150398254, -0.05507318675518036, 0.3485032021999359], dtype='float32').reshape([6]),
            paddle.to_tensor([0.02018606662750244, 0.3696957528591156, -0.27263444662094116, 0.12233604490756989, -0.07948026061058044, -0.044232383370399475], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_4a5f9b31d1d503df34831c5dceefaad9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff10856afc23f7aff18e2b675cc93008
    def get_inputs(self):
        return [
            paddle.to_tensor([0.04015878960490227, 0.8693994283676147, 1.2126078605651855, 0.40155017375946045, 1.172589898109436, 3.6748974323272705], dtype='float32').reshape([6]),
            paddle.to_tensor([1.040158748626709, 1.8693994283676147, 2.2126078605651855, 1.4015501737594604, 2.1725897789001465, 4.674897193908691], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_ab5432dab3940cce009941660fdae273(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab5432dab3940cce009941660fdae273(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1744, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e25415c4b1a3a75ca7139fc46063b40c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(66936.390625, dtype='float32').reshape([]),
            paddle.to_tensor([0.34087127447128296], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1bbcb20a0426b717538338f1c8102a3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(102946.6015625, dtype='float32').reshape([]),
            paddle.to_tensor([0.34087127447128296], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7ea148f35d5447485a8d142894ffdaf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(946.1704711914062, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f6d26b909ba0a914711aa1e61eeca30b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([8400, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_81763756a4de0a896f491ccd483c5ee3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9f53ed43e183f6f02e9e10622b37f255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d7847848f8dae9ca754775f32e227052(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7364055f7814905dc4901270efaeb86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e860f7e60c4691600cbf729668362435(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e860f7e60c4691600cbf729668362435(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1547, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc81189f639892f4b73c40e71b05f56e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(223946.1875, dtype='float32').reshape([]),
            paddle.to_tensor([0.03846472501754761], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_48b5c37fa3ceb4b1926cbbcd5be62bfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(87849.6875, dtype='float32').reshape([]),
            paddle.to_tensor([0.03846472501754761], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7c1434049fb6e4a897940d014a467390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24327757954597473], [0.24460747838020325]]], dtype='float32').reshape([1, 2, 1]),
        ]


class TestPrimitiveOp_f69ac909d4915371a12db1de0ee05cc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1d6ebd010a5bb57e1b0bc1cf9a08133b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b57f68b508afb0c04a8e20d98df7bfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.08816822618246078]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_fcf89e937ccec60284d69dc4f5065344(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.061989568173885345]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.15015779435634613]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_f8a2359f20b3d26249f78cd479b6c902(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.09975819289684296], [-0.0024545681662857533], [0.05102035775780678], [0.03848220407962799], [-0.038630466908216476], [-0.04866786673665047]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_51bbfa7350c2bb06815a4ddea95da183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1255517601966858], [-0.0025407709181308746], [-0.02640495076775551], [0.020489811897277832], [0.03804345801472664], [0.06210698187351227]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.025793571025133133], [-0.004995339084416628], [0.02461540699005127], [0.05897201597690582], [-0.0005870101158507168], [0.013439116068184376]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_53225a031ed4fd6722977c834b118847(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8917ad1a8e25a5828d2c9368e7032572
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24739119410514832]]], dtype='float32').reshape([1, 1, 1]),
        ]


class TestPrimitiveOp_d3c47f25e9b4fb864c73fcc4b446a37c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9871bf1396c8b9db33c6571fcb21ce8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_309f51c50c9850cd16dd3894e8a41433(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85bf86a909bb1d68e72e0ac0c61725c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(60.208553314208984, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_b3f83ca57f9d95e9073a765aa16d0036(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(553.773193359375, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6af6421b9c12b73e1afadd988115b365(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6af6421b9c12b73e1afadd988115b365(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2056, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3e8e8283cc1e6886da91f9e4e11d210(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-49819.81640625, dtype='float32').reshape([]),
            paddle.to_tensor([0.4009269177913666], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bcf1dffc21b330f6ce7333de317a1f6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(116636.6953125, dtype='float32').reshape([]),
            paddle.to_tensor([0.4009269177913666], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_fd25ef24ed252b29527539cdc530dff4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81763756a4de0a896f491ccd483c5ee3
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9fd93520018158bb64f4062d3c180d28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e8cc3e0b780c173af3c98bd79282642e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2228458672761917], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_ce8b7674bf2ad02bed13243898801594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce8b7674bf2ad02bed13243898801594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4650, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_675067de48127f593830cfdeff1ef736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(118754.421875, dtype='float32').reshape([]),
            paddle.to_tensor([0.05398424342274666], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c9b13a8a69c975587e1ec18b65f1a2fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(264429.375, dtype='float32').reshape([]),
            paddle.to_tensor([0.05398424342274666], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e1f386874fe7f2f1d5a272dec62f2487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_703b9c4a7e8742899c426a973f3f855f
    def get_inputs(self):
        return [
            paddle.to_tensor([309.3348083496094], dtype='float32').reshape([1]),
            paddle.to_tensor(2434.0, dtype='float32').reshape([]),
        ]


class TestPrimitiveOp_bd7d4a30c774d0727360587712dfcbb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd7d4a30c774d0727360587712dfcbb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1059, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_feae537cd2cfbe3b5ae03078c0c6d02c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-100001.078125, dtype='float32').reshape([]),
            paddle.to_tensor([0.43673697113990784], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e3195262242ee396e09a2dacd551d15a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(15101.998046875, dtype='float32').reshape([]),
            paddle.to_tensor([0.43673697113990784], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cb7948835bad5bd55157e3be8dc9c037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2efa3b98bfee3abd25f1777188941b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e3e1eb20d931debeaf5847bf437d16f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16913844645023346, 0.4727778732776642, 0.10758273303508759, 0.37166494131088257], [0.18510016798973083, 0.1757633239030838, 0.25953125953674316, 0.05246451869606972]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.27722370624542236, 0.1890515238046646, 0.33570772409439087, 0.22562848031520844], [0.1832714080810547, 0.02770002745091915, 0.048722028732299805, 0.25613683462142944]], dtype='float32').reshape([2, 4]),
        ]


class TestPrimitiveOp_9c80361c71d4987f902260c9caae754f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac9a95d3deba020a96ca1058630f72ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6069, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_279e12a9671d6eca5152897f5a55fcbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cf226b2b1e136d7d303b37d86cea4c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.29850247502326965, 0.1304519921541214, 0.40336889028549194, 0.17085304856300354], [0.1957167387008667, 0.2721782326698303, 0.4003048539161682, 0.3005819320678711]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.16436350345611572, 0.4645549952983856, 0.27595949172973633, 0.004646186716854572], [0.16020409762859344, 0.2444888949394226, 0.45831000804901123, 0.4200648069381714]], dtype='float32').reshape([2, 4]),
        ]


class TestPrimitiveOp_ec74891936ec8616a91a02a2d78d3058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.12368782609701157], [-0.034869760274887085], [0.012797508388757706], [-0.07258497923612595], [0.003080888418480754]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_e1fa1c5e975f8033d25eb9ed13c08c6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12689360976219177], [0.011181898415088654], [0.0017336132004857063], [0.09379806369543076], [0.021462561562657356]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0032057890202850103], [-0.02368786185979843], [0.014531121589243412], [0.02121308632194996], [0.02454344928264618]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_f83df8c4a4c7660376a3aaaff4657c15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15872637927532196], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_e4e8ccbae22d55caec42f8915a479c02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcc39eb9f4f10134caf76738aca0515f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_344a733430d0cd3a35616acc0d219c23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9bdf48675cf4a7b265c061517f9d60fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9bdf48675cf4a7b265c061517f9d60fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2347, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab20a3e7b5c463b7536d94f0f50c1660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(178268.25, dtype='float32').reshape([]),
            paddle.to_tensor([0.15908879041671753], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c76745811fe15af624e47be1d61e0f19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(133584.921875, dtype='float32').reshape([]),
            paddle.to_tensor([0.15908879041671753], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_2aab50715d52ebb4ad6c6cfd69c9a581(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2aab50715d52ebb4ad6c6cfd69c9a581(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3109, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7c1fe09ac90c66b8e7e6a4a4e492006(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-193438.875, dtype='float32').reshape([]),
            paddle.to_tensor([0.08062808960676193], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1db1d676514c64473880df29d971c66e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(176941.484375, dtype='float32').reshape([]),
            paddle.to_tensor([0.08062808960676193], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_74f0947a529b94269eaf441167c3c280(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_74f0947a529b94269eaf441167c3c280(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3813, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31b1a623a74e6cd93e07c51f4891315b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-385659.65625, dtype='float32').reshape([]),
            paddle.to_tensor([0.1846206784248352], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6092fdbde0ba8d81d92aa645a553d69f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(216353.765625, dtype='float32').reshape([]),
            paddle.to_tensor([0.1846206784248352], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_f2b19ff3f9d54b34091d2e852bf2a028(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4644785523414612], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_9fd93520018158bb64f4062d3c180d28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c72522e9333ec7373f72b446df2fc86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(15.829094886779785, dtype='float32').reshape([]),
            paddle.to_tensor([3.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_891fc3a0ea6ba9ba9002c2f1ddd3bb69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([20267, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([20267, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f59d231fd8032f63e5a28d7886df1d0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.010650492273271084], [-0.027310103178024292], [0.0060086315497756], [-0.07410187274217606]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_7fb5d709c2af51a8b2e9161b11b3f6fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.00317885959520936], [0.034122269600629807], [0.015756379812955856], [0.04364442825317383]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.007471632678061724], [0.006812167353928089], [0.021765010431408882], [-0.030457444489002228]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_09e93c994babc934c6d74fddd1cff7f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(4.7184529304504395, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_42d610ec0db72f4ccc37c47f6a6102bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_42d610ec0db72f4ccc37c47f6a6102bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe46004da8f8a18d8e9ca0590b9efdba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-147975.78125, dtype='float32').reshape([]),
            paddle.to_tensor([0.33950138092041016], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7e3ecaba8aaf053ba70742b6dd04b9f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(29830.349609375, dtype='float32').reshape([]),
            paddle.to_tensor([0.33950138092041016], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cc90e638cee2b1bdf7601d7733dc29f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba09068dcaa350fe7e978f1a7a395d1b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3480321764945984], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_8669be9a990ef9dbe0cf85d3ce0ea6de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(35.91263198852539, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_db88b631d7f5d0d057e74c91c8b830fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6804, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_344a733430d0cd3a35616acc0d219c23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_800efd5862dd852986afd1725dd203a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(236.4272918701172, dtype='float32').reshape([]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_7fd3655619af3dbbe2ac1f3ec82009e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(133.09878540039062, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_cc712d615bb205a08ac859dc5b80a517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1fb6aed1d3ff583ef33396027c0c1636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce3cfe8d8b373deb2997124b90bae2d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce3cfe8d8b373deb2997124b90bae2d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc8d6c68d3333f8ec2b05df6196e90e6
    def get_inputs(self):
        return [
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4231, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b8f0de5a7d28b78dbe89c8259227fa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(-984304.8125, dtype='float32').reshape([]),
            paddle.to_tensor([0.037905603647232056], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_bfc464da77376e4f84e7247cc8f4fede(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7c07c53a0fe3540238159515a9fda69
    def get_inputs(self):
        return [
            paddle.to_tensor(240665.375, dtype='float32').reshape([]),
            paddle.to_tensor([0.037905603647232056], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_1fb6aed1d3ff583ef33396027c0c1636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b8602b80e5c8f77f900d18a5a19942
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()