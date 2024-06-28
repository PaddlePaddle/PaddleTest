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


class TestPrimitiveOp_4faa757fa37501bc704ca309908ebe26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5370c591ed1e0aa548b17aed6dfa997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_143180e5bb03c768cc2c58ef5ea70a0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_6be9389ff3108f5526181f1401934842(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a25f59d5ff10c8f801cde947556a0a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.9741604328155518]], [[3.4200820922851562]], [[2.311586380004883]], [[1.7036762237548828]], [[1.4966416358947754]], [[1.8336299657821655]], [[1.8902963399887085]], [[1.966781497001648]], [[1.721276879310608]], [[2.1930387020111084]], [[2.0948421955108643]], [[2.9380152225494385]], [[2.3448522090911865]], [[2.0822067260742188]], [[2.8329405784606934]], [[2.694319009780884]], [[2.0463781356811523]], [[1.9831814765930176]], [[2.2545249462127686]], [[2.056878089904785]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_214f3fa71e396cd34f64d750f27cd034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_d64b540215a8fd38f602e8469d511f0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
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


class TestPrimitiveOp_a45b06b9dfacd25c01281bd71ad06c15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
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


class TestPrimitiveOp_c5370c591ed1e0aa548b17aed6dfa997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd4d940241fd49dffee6b49d82e65b52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d64b540215a8fd38f602e8469d511f0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7dbae266d0f3fea666780391b39dd2a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a78a85e9c03c839d38267b763f0b7a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
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


class TestPrimitiveOp_00baa0448a55e76bcf0807a7570ccc8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b2b8eac10f81a8a7e06b68f3ae0b2db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c4a999274010f1ce4509a2a6517443b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
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


class TestPrimitiveOp_87b0622a897c11ca6c6d31c12b1509ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af38ad4f67aa4c349111a9eed435cc6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_39711661d91ac3a1f5a3bfa2805f6c30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_c5370c591ed1e0aa548b17aed6dfa997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_3e1dc809117cd9d70178706694b1e5c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd4d940241fd49dffee6b49d82e65b52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b563e4e272870ae23ba6d878834d14c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_cecc540604e6daae03ea075ad670c820(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d890ab4e4909148693051ab642c2917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_be1607bd82ec310bafa3b56fe87f6072(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_294d73263cda16af127b9eb173735624(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1fc55b7db8cc87ececb8c58733cdec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_be1607bd82ec310bafa3b56fe87f6072(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_0496e1bcfd0742da59020f84f05658cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1df9af363213018956d3ea3631b1a984(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
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


class TestPrimitiveOp_3c59e9e9c20248feeeee49d8f64a11a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_fb45f7640c191537bd601d17b9166680(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d64b540215a8fd38f602e8469d511f0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_77b9d196e4a3ff9cd0462a183fdc37c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4eb39545333465dcfa49f95a7698905d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_294d73263cda16af127b9eb173735624(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd4d940241fd49dffee6b49d82e65b52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af38ad4f67aa4c349111a9eed435cc6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69690d45f3bd19de56e92929c018be79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dee4dc40266352744b2bb5adeed7ed82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_943f6db2b5c936f3cc36c2d68b596979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_355cd1c25e441a76be4d7451a8fc9717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd6a10137d23dc664f92a5e5fc01abea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2de03032993e304f0a268202d7335600(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_bf1dd20d4978f199442ef45606be94ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
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


class TestPrimitiveOp_4eb39545333465dcfa49f95a7698905d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd6a10137d23dc664f92a5e5fc01abea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_77b9d196e4a3ff9cd0462a183fdc37c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_fb9bf6144e2e2cb403ef792e4eaa31d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
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


class TestPrimitiveOp_ee12127d1dbc3efd005013d8b461ead4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd6a10137d23dc664f92a5e5fc01abea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_be42a4ef7208a2028bfe3257054afc7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70731570c982a420f5efd15eaa8f2645(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
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


class TestPrimitiveOp_dccd8e3e797efa6aa8983e17a164a56c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c59e9e9c20248feeeee49d8f64a11a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f4e3ece847acc5bb00d5e78fb6d6a2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e10d3d5d394ad866ea75d0bd5c35b70d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_1277b59bab51ee9855248ba0115a6a76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
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


class TestPrimitiveOp_3c59e9e9c20248feeeee49d8f64a11a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cecc540604e6daae03ea075ad670c820(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1b2b8eac10f81a8a7e06b68f3ae0b2db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69690d45f3bd19de56e92929c018be79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af38ad4f67aa4c349111a9eed435cc6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_77b9d196e4a3ff9cd0462a183fdc37c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_56ed732cd6ae628db8664df952fbb4b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8760808aa480c2ee869a9dc62b581903(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_203a06a3fb2319d747ec8c10bfbf341e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_8760808aa480c2ee869a9dc62b581903(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c5370c591ed1e0aa548b17aed6dfa997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d64b540215a8fd38f602e8469d511f0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb75353f67a77d48a659929ae326f3a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_bb75353f67a77d48a659929ae326f3a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_355cd1c25e441a76be4d7451a8fc9717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee12127d1dbc3efd005013d8b461ead4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_441cee4406c3ddabdba98422a595db42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.4739737510681152]], [[2.2430005073547363]], [[1.9302268028259277]], [[1.719527244567871]], [[2.5277698040008545]], [[1.271949291229248]], [[2.0438966751098633]], [[1.9506549835205078]], [[2.3556389808654785]], [[2.778470277786255]], [[3.0379576683044434]], [[2.195894718170166]], [[2.291151762008667]], [[2.030898094177246]], [[1.5601255893707275]], [[2.619415283203125]], [[3.070551872253418]], [[1.4371898174285889]], [[2.272088050842285]], [[2.9633190631866455]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_214f3fa71e396cd34f64d750f27cd034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_976976572868e2c153cd3e4cfa0b1ec0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_527239628aa1284706996ba5a7f3e186(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5588444a02db7287cfb6493ba632d9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90a5309476f3fd7dfed2badc04f53507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b151eedb2ac10a609d14468fd3af4718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4eb39545333465dcfa49f95a7698905d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_569595f1e16a690800ca4940fb8d73cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92299c1da8a1416abd2879e8b7f0469e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e887ad0dfd05e82ce4e48dbd19e9b37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_16f06ca14ba90f0db7daeefb499cc0b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1fc55b7db8cc87ececb8c58733cdec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_9a003115d9db55ade72b0de7f39ab8b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.0652778148651123]], [[1.7512539625167847]], [[1.9886212348937988]], [[2.221742868423462]], [[2.8262486457824707]], [[2.085285186767578]], [[0.6642834544181824]], [[1.1961097717285156]], [[1.2067431211471558]], [[2.219651699066162]], [[2.3536672592163086]], [[2.0006103515625]], [[2.327887535095215]], [[1.6601722240447998]], [[1.5111178159713745]], [[2.230372667312622]], [[2.0499329566955566]], [[1.5385329723358154]], [[2.1554510593414307]], [[1.6767657995224]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_214f3fa71e396cd34f64d750f27cd034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_976976572868e2c153cd3e4cfa0b1ec0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90a5309476f3fd7dfed2badc04f53507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a45b06b9dfacd25c01281bd71ad06c15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4eb39545333465dcfa49f95a7698905d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c59e9e9c20248feeeee49d8f64a11a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_355cd1c25e441a76be4d7451a8fc9717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_943f6db2b5c936f3cc36c2d68b596979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_24cea99ec99131ecbbafa9689fad7719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0c2bd1206fd47fcd1a6eb5b433ea6270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_b1fc55b7db8cc87ececb8c58733cdec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_f460e0e69f5781c1c827ac5b2dda2b35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14811e48d9e1f10e4dd5dc15df655366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_00baa0448a55e76bcf0807a7570ccc8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dccd8e3e797efa6aa8983e17a164a56c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d890ab4e4909148693051ab642c2917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_102eac7f6980b468112730aaf55af521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dee4dc40266352744b2bb5adeed7ed82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9cb148700d96d897494fc63afbed0d78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee12127d1dbc3efd005013d8b461ead4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ee12127d1dbc3efd005013d8b461ead4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c483e7b7cb6bf702dc74ceba7075252a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 7, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69f4e9e920e18b0a8a3f40b74c858ee8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf595e527e5dcaf0a62646bcb51d2791(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_77b9d196e4a3ff9cd0462a183fdc37c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
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


class TestPrimitiveOp_03df105ee8b9cb63f686e3c78001c62f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7dbae266d0f3fea666780391b39dd2a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9cc08f9f581085abecb172cb63992b4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()