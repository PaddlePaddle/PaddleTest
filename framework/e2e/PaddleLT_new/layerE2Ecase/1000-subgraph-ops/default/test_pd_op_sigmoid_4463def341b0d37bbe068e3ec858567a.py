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



class PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e10d3d5d394ad866ea75d0bd5c35b70d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_50116cd8ba74b4afa853fa832acce035(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4cffc6b6960da3166bf91b1d593e1ef4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_81512dd4663279440e684d42a665eca1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_987c13215081c99a00b5dd2a43448e97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81512dd4663279440e684d42a665eca1
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_825bce5a071a85403d3c7a41b536aaee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 84, 84], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec4c67dcc738e5c5b9380f0585041a0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bd7ebcf3dd30fc94fae0783b93061ba5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06f2f46a5195a55783fe01fd0e1d64ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc594d73cecca3665b2f4ed6ba411419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ef574d040d0ddcfcb895bb3bf21dadb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 960, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5eab2615ce8ce249a1e0d751e22596dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef574d040d0ddcfcb895bb3bf21dadb7
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c16888b0c6c48c06b635c98ce0fa0a0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 15, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_172eff3ef14b26044fd75bcf50409caf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2bd3c9d81df46b3fddfe935a470e65d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9396564ac8ba422dc6c037e6301ce5a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bd3c9d81df46b3fddfe935a470e65d0
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.596120595932007]], [[2.2330002784729004]], [[1.5223428010940552]], [[2.015205144882202]], [[1.9397326707839966]], [[2.2572758197784424]], [[2.1204748153686523]], [[2.422625780105591]], [[1.6845693588256836]], [[3.138563394546509]], [[2.6652865409851074]], [[2.0771877765655518]], [[2.8043019771575928]], [[3.0215392112731934]], [[3.0406367778778076]], [[2.653716564178467]], [[1.8741563558578491]], [[2.819385528564453]], [[3.0009822845458984]], [[3.5632433891296387]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class PrimitiveOp_179a8ba108d2d1d0b1cb96508d4add60(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_14e388fc29e3c4c0438020c84a37b008(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_179a8ba108d2d1d0b1cb96508d4add60
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38799c5591394d315293317330b5081d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01de5ac65d16a78453b7296d6db60b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_30660460f49618355c95e4c137ca227b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_19e2cf5d3a2ae65e84a78fd106e97f7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30660460f49618355c95e4c137ca227b
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_96362313c17c00d9b82a622db0d3ac92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0845e64f0ce71b3955ccc3d6b905f593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96362313c17c00d9b82a622db0d3ac92
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b563e4e272870ae23ba6d878834d14c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_987c13215081c99a00b5dd2a43448e97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81512dd4663279440e684d42a665eca1
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb70c80c6b7389ab5c9382382265bc7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_19e2cf5d3a2ae65e84a78fd106e97f7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30660460f49618355c95e4c137ca227b
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f520d176aa33d744a76b643b423a1df2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1152, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0edea2b16b1a28176e74b606b808608d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f520d176aa33d744a76b643b423a1df2
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f5ac8d41b453cdd6082848f2d95847d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96362313c17c00d9b82a622db0d3ac92
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b0f47dafb082a1610550987800c8db0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b973809b64721a97043a28147ad9a800(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_578eae2cf8cdf16c0d514e6dfcb8261f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_327a3a701fd04a6a3c6ad035a4389eaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_578eae2cf8cdf16c0d514e6dfcb8261f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a59a79268a95fcf8453204ea5fa51ed4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96362313c17c00d9b82a622db0d3ac92
    def get_inputs(self):
        return [
            paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d8b49a67b51d882597273db86b281a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_62fe84f289c257ce5314485f146926b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c3d8f799058089b9c998a69c46893db3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 15, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_53d1bf5c0537beff9be392ad472a8a9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3d8f799058089b9c998a69c46893db3
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0e416db2ec316af29de976716ea40ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6301b8cb3463fc82a8e8cd61c6ac3b41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_569595f1e16a690800ca4940fb8d73cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_19a2251f5a5183413595026de5fee814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_2a99da29e4a71689ab635b933d4c7cce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6a73c3eb02fdbd0ba6d064c50a98445d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a99da29e4a71689ab635b933d4c7cce
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8f75837ed0b342f9908d05e6cf957b97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 76, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06f2f46a5195a55783fe01fd0e1d64ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1680665f7c3bd5ee5308470d36126e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_987c13215081c99a00b5dd2a43448e97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81512dd4663279440e684d42a665eca1
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_745123c1708a9ade5a7df929c152deff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_76a7e8117b403fc8d354e0136baf6972(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2b3cb6e670b1d3f8808a1f396b1a8ef5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3d8f799058089b9c998a69c46893db3
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb70c80c6b7389ab5c9382382265bc7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_527ae733035c0b6cb324c304917a8be9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3d8f799058089b9c998a69c46893db3
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6301b8cb3463fc82a8e8cd61c6ac3b41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_093e25e837cf5103b80b458d09705142(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad6f253ee93021e8425e1d5a7f452538(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_73239ac899523a3cf15cfecea4ac0ebf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a3781b682968f65f9f5c1f35cbd61164(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73239ac899523a3cf15cfecea4ac0ebf
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_415ee9cca9126e66f822a106eeeaa04e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_27308b4b91b295c5a40d5f6262b91893(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_926f6e1cff17ddcae895625d06a0de4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27308b4b91b295c5a40d5f6262b91893
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fafcad5e98315128a089fdf3d7c28e9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad8e00d2da88a86c2736a55490885700(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_415ee9cca9126e66f822a106eeeaa04e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c86abcef9840bf0a32b5de3ea6266cb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc594d73cecca3665b2f4ed6ba411419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cef5cffca9021c6f33141299c8cec5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 21, 21], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_19a2251f5a5183413595026de5fee814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_634cbab2d7af7c90dadc72b6d0c6954f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 34, 34], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e84bc6c3c29fd89a63c38dca528664a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_927746f3d42310dc9fa75d10b0938f80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96362313c17c00d9b82a622db0d3ac92
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6258da6093a4f04ec706ecd7a7a084f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fbcee2f666ff326a2b9cbda6754f9322(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 18], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9d57792a62e3db18b316a609e9d3cfbc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ec9f97c69cedb1614721260a45f35acc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d57792a62e3db18b316a609e9d3cfbc
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb554d6872cf5aead575530ed9d4e5da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4e264aff28550d5b987e733e31f6b09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 17, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_53aca09af1bf1031591a5d021a60c829(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fb45f7640c191537bd601d17b9166680(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_19e2cf5d3a2ae65e84a78fd106e97f7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30660460f49618355c95e4c137ca227b
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_778316f0c401ef55320d8753bc2b7437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b563e4e272870ae23ba6d878834d14c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a5dfcccd5aae3f0ba6bb9ac661c7c45e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9601d3d4b916e14334448332d6e50ce7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d57792a62e3db18b316a609e9d3cfbc
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cea6de878311e3e4e3fad4f246fafecf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81512dd4663279440e684d42a665eca1
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_926f6e1cff17ddcae895625d06a0de4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27308b4b91b295c5a40d5f6262b91893
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb70c80c6b7389ab5c9382382265bc7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0e416db2ec316af29de976716ea40ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edf4c333e724a46c0b40f0d2a69d8643(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f520d176aa33d744a76b643b423a1df2
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_0011b5e9c1861835befd0241647f83db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a920420f11ca363139928e53eb431821(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0011b5e9c1861835befd0241647f83db
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11109ce159a4e3b57cfe319931361eea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb554d6872cf5aead575530ed9d4e5da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58f9e28a5c4061886f07dc07aa848afb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27308b4b91b295c5a40d5f6262b91893
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4ac4dbaa77b53614fce28768ddb693c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21f31c875d8532efcdb1f34bf9f1fcfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_3e03067fc6fa2402695b0d2e270b7472(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_da2d57a3cc7090498f307c337389466a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e03067fc6fa2402695b0d2e270b7472
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65340b489f3be48e1fcac4c138cd7a0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 46, 46], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e8aafeb52858fec43004294803467176(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96362313c17c00d9b82a622db0d3ac92
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e10d3d5d394ad866ea75d0bd5c35b70d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cea6de878311e3e4e3fad4f246fafecf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81512dd4663279440e684d42a665eca1
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21f31c875d8532efcdb1f34bf9f1fcfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9601d3d4b916e14334448332d6e50ce7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d57792a62e3db18b316a609e9d3cfbc
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e10d3d5d394ad866ea75d0bd5c35b70d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bae922cfe1c00b607119d1f2a4edfe18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96362313c17c00d9b82a622db0d3ac92
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6399cf612c3c2b03211eab83e20f576b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8606a0e0419e2c6107040722f6bdb20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30660460f49618355c95e4c137ca227b
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_21f31c875d8532efcdb1f34bf9f1fcfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd6152129a483c1e485057aa65dd7a32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 60, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79341a20be09cbbf802b9cff4bece64c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f6a29922e397ef55faf130401453914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96362313c17c00d9b82a622db0d3ac92
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86c7e0d98d23c4a6129d89434e70f7f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96362313c17c00d9b82a622db0d3ac92
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e9a1fb49d040c89ae4dc92500ec1b37a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 96, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffd61dafabe54cc65b91dd06228ac057(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_62fe84f289c257ce5314485f146926b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32af590d25b6870988a0996999cd482a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0011b5e9c1861835befd0241647f83db
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec9f97c69cedb1614721260a45f35acc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d57792a62e3db18b316a609e9d3cfbc
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_365d4602ed37863845d2d5cb045963f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_069459138c2b70483d41b8fbc98d4927(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_365d4602ed37863845d2d5cb045963f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f9dabce83cdb7c055c287983bab2945(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3d8f799058089b9c998a69c46893db3
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6399cf612c3c2b03211eab83e20f576b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bee050a73bba614fc9648240cb3996e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 80, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6301b8cb3463fc82a8e8cd61c6ac3b41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b0f47dafb082a1610550987800c8db0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b0f53d6c9cf976d4df6d2040b0e257a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 23, 23], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b563e4e272870ae23ba6d878834d14c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_569595f1e16a690800ca4940fb8d73cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_569595f1e16a690800ca4940fb8d73cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba6613556de91a08c853d82175bbdf79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96362313c17c00d9b82a622db0d3ac92
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad8e00d2da88a86c2736a55490885700(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_74ff25d3946f91dc0ea061a15915cd9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_19e24007ee31e64f15ba995c25939525(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e10d3d5d394ad866ea75d0bd5c35b70d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec9f97c69cedb1614721260a45f35acc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d57792a62e3db18b316a609e9d3cfbc
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ad6f253ee93021e8425e1d5a7f452538(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_327a3a701fd04a6a3c6ad035a4389eaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_578eae2cf8cdf16c0d514e6dfcb8261f
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edf4c333e724a46c0b40f0d2a69d8643(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f520d176aa33d744a76b643b423a1df2
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0e416db2ec316af29de976716ea40ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9601d3d4b916e14334448332d6e50ce7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d57792a62e3db18b316a609e9d3cfbc
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_23c48fe2ea99c7caa311043960b843b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc64c0fb7191aeaa189e29e24a3c42ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_627efa1fbf84f3f6524a8790b2b6fc36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff290c602369f0fddc0316aa586b175a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79a51bc64df40d53793a11d6884f7336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 192, 288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9b0f47dafb082a1610550987800c8db0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85cdad7bebd3d48be1db299245473284(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0f59ab6b5b8d957c483a702345199924(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 68, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ff290c602369f0fddc0316aa586b175a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_987c13215081c99a00b5dd2a43448e97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81512dd4663279440e684d42a665eca1
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_19e2cf5d3a2ae65e84a78fd106e97f7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30660460f49618355c95e4c137ca227b
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d302bd203f7733f6268a040c04b43d12(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8e308015cabd6b9f6d1e7ccb28ab1d5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d302bd203f7733f6268a040c04b43d12
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7a977182eb7c8a86b339aa35ef627d37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2a89c86afb1dee139a35b70b715ab15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 72, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_33f343538345e701b879300c5b6b7aab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 72, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e26d3a126e8fcc610b52c32f0482ad6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8e308015cabd6b9f6d1e7ccb28ab1d5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d302bd203f7733f6268a040c04b43d12
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b900d5d2ed0b1400f24ed107d2e515b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4ac4dbaa77b53614fce28768ddb693c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8606a0e0419e2c6107040722f6bdb20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30660460f49618355c95e4c137ca227b
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06f2f46a5195a55783fe01fd0e1d64ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_446526674c1e45fbbe5dcaef803923c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1f7563d024b4ddc25e415667e0be6bdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_446526674c1e45fbbe5dcaef803923c0
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.9604312181472778]], [[2.298386812210083]], [[2.349522352218628]], [[1.7564992904663086]], [[1.7989057302474976]], [[1.964356780052185]], [[1.9731441736221313]], [[2.2705907821655273]], [[2.160538673400879]], [[2.490543842315674]], [[2.2945728302001953]], [[1.785688877105713]], [[2.4100825786590576]], [[2.2321295738220215]], [[2.777137041091919]], [[2.3318593502044678]], [[2.71602201461792]], [[1.7619919776916504]], [[3.0053598880767822]], [[3.0894699096679688]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class PrimitiveOp_13203893eee5872df8e2a5c90635fb7b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1aa0a12f2813bafc0b06141626c762f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13203893eee5872df8e2a5c90635fb7b
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a6c3a2868e660a050e72bdf7f9c33c2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afc98df9019036dc717275780b3bd305(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6c3a2868e660a050e72bdf7f9c33c2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_50005db0bcc73b7b06e2fa463ddbb52f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c848b2b48d0b47676a742a076cf105c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50005db0bcc73b7b06e2fa463ddbb52f
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db6cd5213cc42f579dec07a369a0d806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72cc4ccab866b98f6ff314de3db1876a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73239ac899523a3cf15cfecea4ac0ebf
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c34cc5f625bbd4830d0ba7c7ab50312(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cea6de878311e3e4e3fad4f246fafecf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81512dd4663279440e684d42a665eca1
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_62fe84f289c257ce5314485f146926b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_abc51ab38d9061764d5426cb2dd4141e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3d8f799058089b9c998a69c46893db3
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_d0a7f87e4c53ce0c1d89572e68684c48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_89e328b7858392b532a3368c895978f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0a7f87e4c53ce0c1d89572e68684c48
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_06f16115d94abe8ed9dc5fe1aaaccb70(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1bfb9f9180107267d0585a9fc22258f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06f16115d94abe8ed9dc5fe1aaaccb70
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 120, 200], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_093e25e837cf5103b80b458d09705142(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0c30a6b171c30e89493c3edab25c1b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 20, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc49b60217505fd156d34cecdae3050c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 92, 92], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_11311160e1d01b6ca0ca3b1d6c225757(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e7e569e51c9e196acdc36c50f38e0cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11311160e1d01b6ca0ca3b1d6c225757
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fafcad5e98315128a089fdf3d7c28e9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_569595f1e16a690800ca4940fb8d73cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_850f91c4b969f45ccbefbf317339f461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bd3c9d81df46b3fddfe935a470e65d0
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.635324478149414]], [[2.4752397537231445]], [[2.87604022026062]], [[1.3973051309585571]], [[2.9939796924591064]], [[3.275991678237915]], [[2.2748074531555176]], [[2.073307991027832]], [[3.05397629737854]], [[2.6619248390197754]], [[1.96149480342865]], [[2.0161914825439453]], [[2.0761265754699707]], [[1.8466088771820068]], [[2.3124935626983643]], [[2.523139476776123]], [[4.1475934982299805]], [[2.6560111045837402]], [[2.8013551235198975]], [[2.962149143218994]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_14e388fc29e3c4c0438020c84a37b008(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_179a8ba108d2d1d0b1cb96508d4add60
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ace34fbf90cd8e7d2ae0b15597c43534(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab6b9fcfd2ee64f82c5218cafd3c0517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ace34fbf90cd8e7d2ae0b15597c43534
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72cc4ccab866b98f6ff314de3db1876a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73239ac899523a3cf15cfecea4ac0ebf
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0845e64f0ce71b3955ccc3d6b905f593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96362313c17c00d9b82a622db0d3ac92
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cea6de878311e3e4e3fad4f246fafecf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81512dd4663279440e684d42a665eca1
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec9f97c69cedb1614721260a45f35acc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d57792a62e3db18b316a609e9d3cfbc
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f4ac4dbaa77b53614fce28768ddb693c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b563e4e272870ae23ba6d878834d14c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1d0cbd4b04200cd4a807fe162998e8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d081658cb7ab1b46de5f4af26597017b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e5a0aebbdaa2274f93ddaa1e9dae0a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58f9e28a5c4061886f07dc07aa848afb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27308b4b91b295c5a40d5f6262b91893
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_41ab67520ad3e907f5de757dc030263c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_58c349290813c24364fd64e137d24615(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 576, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b0de89fcd344c03af1c697eb0c9ae4d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58c349290813c24364fd64e137d24615
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3505db84ce26720f7f815b5414ea67c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 42, 42], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fafcad5e98315128a089fdf3d7c28e9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_77741bc7aeef1704d1e1d7b65f0bee32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfb1846346765aac8d143ed9b4f068b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96362313c17c00d9b82a622db0d3ac92
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_6f473609c821fa63166dd343d16b6c2a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_893676b2517be8893891f4cd70b2bbca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f473609c821fa63166dd343d16b6c2a
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b973809b64721a97043a28147ad9a800(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_32af590d25b6870988a0996999cd482a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0011b5e9c1861835befd0241647f83db
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3781b682968f65f9f5c1f35cbd61164(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73239ac899523a3cf15cfecea4ac0ebf
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_38d056320bcd1c579a99d9f92a82b80f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 288, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d470a7ccc11e476faf703d798912ad7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38d056320bcd1c579a99d9f92a82b80f
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a920420f11ca363139928e53eb431821(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0011b5e9c1861835befd0241647f83db
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_193b0dc5eba13d4335033341c81b6189(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3d795ccc70c7e48765fe9e9f4546db92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_193b0dc5eba13d4335033341c81b6189
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8606a0e0419e2c6107040722f6bdb20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30660460f49618355c95e4c137ca227b
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8606a0e0419e2c6107040722f6bdb20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30660460f49618355c95e4c137ca227b
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c8608240d77aa0b08604cab7d218f6b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 7, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_33f38355e8db41f08e260b9740a07825(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8608240d77aa0b08604cab7d218f6b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 7, 10], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_cccda71426ac8a9fd0b81561cc37ee3f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cef302fa15ddf907dc0c108c4acf1392(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cccda71426ac8a9fd0b81561cc37ee3f
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf7f7138027fcfb189de30256bdcf03f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50116cd8ba74b4afa853fa832acce035
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 14, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85cdad7bebd3d48be1db299245473284(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9601d3d4b916e14334448332d6e50ce7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d57792a62e3db18b316a609e9d3cfbc
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc594d73cecca3665b2f4ed6ba411419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1b2b94e3db820581801069494108cfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 48], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_617fdec5cd0b86c85b17013305ce2cb3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.nn.functional.sigmoid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0bacbbe6feb1787fde986fcb40813188(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_617fdec5cd0b86c85b17013305ce2cb3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0edea2b16b1a28176e74b606b808608d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f520d176aa33d744a76b643b423a1df2
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f546a2112a284d13ab2560dd1443c53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06f16115d94abe8ed9dc5fe1aaaccb70
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()