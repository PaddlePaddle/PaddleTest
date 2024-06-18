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



class PrimitiveOp_fe82cab8d58e76c293086a19456b745b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_abd36e134727e1e9f079a0cf8ef9c9a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 176, 176], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8c14f730699b40fb72f10dfa488a7594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 88, 88], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cf3dc3223c8d4061ea3a63b5db844f3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5897d4aaab632ce2794a87c024267bf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b0cca92d23f8a3528351b38f9990f2bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e942205aefce9a1785c0693c637bb405(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 176, 176], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3d84e7de8f61ffb20eda76d1144e8e90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 88, 88], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3183559dad5d5a9cb48d87604a0b6b6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 44, 44], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_499c0d508cc4a63702153e58bdbb2eac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 22, 22], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c6dbbb9cb2978368bd23f88e94279894(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 11, 11], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2244dcb4020c8c125fb6871855641982(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 184, 280], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8507625b0f85c2f4c2bde85ac03815c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 140], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8bd42a1da25f5d951894893c1eaa819(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 70], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_43f755a12fe935d45ccb55b0ecb7d4ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 35], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_487471010425cabb4f22d0606dce1f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 18], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a68190cb2ed5e8235614f002ff350c19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 184, 280], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_144e0366492710a3182bd77653540aa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 92, 140], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f0b2bde537be6de0ae6285867c383b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 46, 70], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e98f1e1f543f63c4de3b20306017c386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 23, 35], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_33a41fff19098d77bfa95ce88f38c6b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 12, 18], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ec48a690c1f8119f96b9b20832556439(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c759c41219231f3fe4eb2fd0289fb98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 3600], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_233eb0681ca9f0ace0526cf3fc7bcf8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 3600], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_136e97252a28207283ebc7c176e337f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_feaaf475a000cb3a5c29f25f7ef81cbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_214effe12236b2021d3fe0660f935895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9b063f68dc51be6775ab9ac44f11fb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 3, 4, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_432cfcba5900e1b1602d6d201fbe59de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_85ccb8abfc50f9aa8814b904f351d9ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7d8407397e7ab85de9129abda84b77a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 8, 8], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b17b2ed7314cdac40106e3df9ee078bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fb3df4c1e1cb841c4c3b9cca3af869be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6da2f0478b691041c8c6be4954f628ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cbd84b4ec7f6f037078b6a0187731811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 7, 4, 7, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9d86c553350cb977b271d6b77a7c3701(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8162403ca90dc35f2c8ef6ee18a49706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 49, 3, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8f4960fb90359e27c95941bff0fd266c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3066931b0d4e2829f9629fbaf746cec0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 400], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e79aef97fe0a6a33f825742a1a77fa9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 400], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_120d39d8330a2e8c2ddd48eba6292911(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 196], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b4980cb50f1097461fc8aede16bbc938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 196], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b7bffe2f20741db9730a36e4cd8d7f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 3, 12, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b4e64f013b433dedb04d9ca8f9edaa43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dcbe50e90b6a8af0940c337561dcf4e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cbd84b4ec7f6f037078b6a0187731811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 7, 4, 7, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8162403ca90dc35f2c8ef6ee18a49706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 49, 3, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8f4960fb90359e27c95941bff0fd266c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eab1004a99a359232d541ba240d9504d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 7, 4, 7, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8a445d1738f7418465254f5deef69fe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 49, 3, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b98150dbb4608341e866819104fbe8ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b4792bb63a1de45334f6d4aae23f3ed6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 8, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_397302b4df49c080289e0023d6deb351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 8, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c3b8ebad3f0bcf76f18c4ca18f9bbaea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3e3a1957fa45f6dbf378722e72f745ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e6923c7c04758b9b93d470c599b9855c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_06dc370c8e6a5d1eb971e690614486df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 232, 16, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_ff3b4a1a7327cf7244031390082a7911(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1561801c215fb2942746211489083f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff3b4a1a7327cf7244031390082a7911
    def get_inputs(self):
        return [
            paddle.uniform([100, 256, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a969cde370e446cf86c1e1fb39609cc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_526f81b2515ed71d19342a23c06ad49d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_052257db646b4e17241106368af3bf99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b2fdf1d719a40598042862b4951e32ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8a149549918a80234878de54c341668e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 196], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f262f2afb18e635df46c59f0c3d7140e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2116], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_63dbecf5821659d6c3d0a6a2817b24fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2116], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_82ba9d3e3673cb47909ed11bb329885b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 3, 3, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5c25b624b82ec758b1db55bce22c2dc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ad34b3b3017bdffd026ab01eaa93dd4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c93a70902343a6481859d47a82215837(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_610ee28a330791b082da8698edf65032(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be81c12378a92aabc5c256aaf2acc56e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 4, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be81c12378a92aabc5c256aaf2acc56e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 4, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0fbe1f86160d358384797459a990b6e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 4, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a3a1d7cf9adc6b522d7110989ab67c43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 196, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1006b823370d28fe0b4e283c48eeec6b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_affea2415ccb081a8b34da3544acd29f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1006b823370d28fe0b4e283c48eeec6b
    def get_inputs(self):
        return [
            paddle.uniform([4, 196], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d52ee771283e4e85919c433001ab02db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1006b823370d28fe0b4e283c48eeec6b
    def get_inputs(self):
        return [
            paddle.uniform([38416, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fbb5ca82b76b30d6c051adec3ba23f0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e33e3592d1920157826667a20b591b4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2a602033d845e512c401616d5b852bcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e9d8c9ed40454a1ef90cdd321d1d1a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 136, 208], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f3a94cc3426a7f6738182b1f35036e01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 68, 104], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6be81b09a51d6e74075bcfbc3a554071(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 34, 52], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_25fad1ca38179ed997eed0ae2e4ae9b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 17, 26], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_121ea095c1eb13165a5ae811ccad5450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 9, 13], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1ca48e9ddeeb24e37b0afaf2a9abd01c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 136, 208], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_512b9c233bdc1114d452a13281792aa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 68, 104], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_882a59826ee31d6acf35b2e00b43a35c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 34, 52], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_196c1f466678bffdcdf08a29744b037a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 17, 26], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_66a75af267fe985169997d91588a0ae2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 9, 13], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b2f598c10084d89b36a7f21475db2cff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 6400], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e91413f0aa91f9d94970187fd9573794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6400], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_25e0f587f492c159c89f777ea1854471(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 6400], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_150bcdd93be65f5f10be2d7d0fe2b6fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 7, 2, 7, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ccb4b9dd51619d0618eca60dc21cc8c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 49, 3, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d115320fa35c0fd07dc09241cb613c74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_89bf0bf84cae872aea0daa45dcf31f98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_af1551d8d99a50ed0981ab50bf4978b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 7, 2, 7, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fffec7177c247744e7c9d03fbf868273(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 49, 3, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5e7c9015cad1acd89bc8315e757b915f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5a2e66c74105364cfe4ee96deec37b7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 38, 38], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b97107cf52e9823a0b50c97f3638efcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 38, 38], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8dcd63d365e3e91fe8118c6ebb730aa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 19, 19], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_763a88e71988c34d5783d4c52fccc0c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 19, 19], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00740363be83bce24a710e46a5d95fc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 10, 10], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_62107fc3511fab7ed75567f2ae236a64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 10, 10], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_327aea06524cacd8d645a89729b6f7fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 5, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f1b71da53babe3abfefdcc95a95b84a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 5, 5], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4b8d18bc74a0efd0a3b9d626582643a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9a5f71184aff201bdb1812b55bd17844(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 3, 3], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_53a642801673d08816d69b18b4e5ce8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.2060820460319519]], [[-0.007826495915651321]], [[-1.5087794065475464]], [[1.2106908559799194]], [[-0.9281951189041138]], [[2.2179782390594482]], [[-0.14296011626720428]], [[1.4449150562286377]], [[-0.7795872092247009]], [[0.838371992111206]], [[-1.998719573020935]], [[1.1525511741638184]], [[-1.2585561275482178]], [[-1.7858591079711914]], [[0.4132246673107147]], [[1.5110154151916504]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_fba3bb639832a1ad6d9bdb395d3cc982(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0827c7220fcbb0aae1e854d6586909c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_414624e7f1fc38d55c6bec9897fc30d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_90fe4ba7b98e5981e555efe36558fb6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a8f2473896e5927ba76170e7641e04b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3cd3e96f9d4a555a99b8ebc27c98e99e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_158a017832e4c4882c6eb9ba110c3374(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([960, 2, 96, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_74004808367195b9016c7f8cf53057cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 1, 1, 96, 48], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d0971aad09db83711ddac16b7c1a131(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1600], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a652f0a00210cb98f838c262c8d1555c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1600], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee48c98c6f5e46916fed737180bc86f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 120, 216], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1f617af2888a8afa8c9abbbc7e9249f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 3, 2, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_41cdfd836fe1e51f13b3fd2afe308ed9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_436f5246aa38d776f9f8ec159162c290(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0827c7220fcbb0aae1e854d6586909c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_414624e7f1fc38d55c6bec9897fc30d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_90fe4ba7b98e5981e555efe36558fb6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a8f2473896e5927ba76170e7641e04b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3cd3e96f9d4a555a99b8ebc27c98e99e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1eff79dd6106d1a4a4fcc4a5c490b231(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([144, 4, 96, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2931cbb8849ce9e5b1740f19522b2e9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([6, 1, 24, 48, 2, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_31c02d7af5e105c8ce7e53b457a2e039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 7, 8, 7, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5cc43026d65da46739995e6c5a5c9e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 49, 3, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e7724832c3cb3806d2d54fb88638a702(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ff8a11cacdced2d1b7ccd6961f0ba93c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5a7312077188274eea5aef46e597a4b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c906333aca84630388ca9fd8cf0aafec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_549f070539b7b40d3b41ebf30e1984df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 3136], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2877f126b0c92bf85ce1b4d812e71b59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 3, 2, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_67f05f393b405d11c2ec0ac45c2bb0a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef5c4430674d8a9cbdff662c9ceb589a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_826b18468967e22575874107f9216b53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f79f03d2f00fec7412adb81db919ccce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_66f31f8ea830cf05a131c5d75e08ccd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e714b5f06353f1294ea7269e166c9036(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dee6f62fea1e2f0743b9ba2f40945e79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([43, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bbdc25bae2bed5beb1557fac7e956403(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3237f091273d4416141ddd70710f2a3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_36f5d0900d125b29a34a266b48387a46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_67a6c5f2246afa78b3604eb6ae97492c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 169], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4cc67ed4669c517ba8f915beeb7be9cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 169], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f5ddd00c809f1c92761050a40b150efb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7144693e26bf7d2aa93348250e6b3a2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5fbd7ea4749ae70e3b2d86fd6d02e2e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eddae9655a680b4db080bfde03b3000b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c91a5dcc94d4075d88a0a0086cc15699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 9216], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_498f66040c57f38ce330066bbb96698d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 9216], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_99899b59f55c43810c72d3c8f2404cc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9216], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54d86df0ce16c6a954ba512079ba85f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ba02d7e3b213a456812462f14d24bb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1968aea9614b59bbef5f1e3729129d2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_41d99ca11f0402148b261ebc6fed2356(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54369f8dac91673b29bf349fd34a6d5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58172ae786745e5d469606360513d15a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([384, 2, 96, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_04657f2dd2b1de832e4133cbd753c047(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 96, 96, 1, 48], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f79f03d2f00fec7412adb81db919ccce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a8ba44956a59533a6b59bad4a06eb53b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 196], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1454d0d5cf9a7b39111f20269dd87294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 60800], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_74498c79da71200187ca04ed185fe5f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1454d0d5cf9a7b39111f20269dd87294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 60800], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b41559a7ba5f9581f5466b9e04863e8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_255647ec46429c20cada9ced99d81b80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_76d6fc9fa1a654ce4c6e3a6611376915(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_42860ff90a7634a9015881c1e068e94d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c82e87b5def4c02b9fdbcbb082def0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([11, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_992021b0b1cf1dc61d3db67922d182b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ad9e2a5c03f70925c5a55778bd5534d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 784], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7d5400be437d5023516c37fa1abf95ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([4312, 16, 2, 4, 6], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8624609f3a54236d85f4f520510ce720(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([4312, 16, 4, 6], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1cbb9f872672454f41f03377d19c5d76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([4312, 4, 16, 6], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83bcb37de7599b0ee6c9e9c8da4898ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([6, 96, 9216], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_791baf5d64cb8faa0a9887a39d48c6f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 160, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_143fc4a55ff5b01e78192fd589e6c7a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 120], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_61ece4317a3c785380bfc716b7f30e66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e13168857939537919d0436f6145cd2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 30], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_63382debeeb01a697e09ed25d6f16aa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 10, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8ba7bf90c23cad0bcef85ff6be61428b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 160, 240], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5eb8f6799748aaa45181a7af4cede837(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 80, 120], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_db977b8d718273256fd1cad1fa6e7ede(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 40, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_37ec06c7d1c71e7226fb2d6bdb03b030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 20, 30], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f5096bb5481afd38d949a222f6e0073a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 10, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8a149549918a80234878de54c341668e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 196], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0cae6e34682e9f8b7b8b4411919f8cef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2704], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_35c0b2926add138e2b1661cdb1288ee5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2704], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_629996cd8510630233d02a3566ad17f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9e7b9c71509cb9f5f30a39a205c326f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 152, 272], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0802e7e5238d8c7d2034425253d5142(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 7, 1, 7, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9851ef260321949b51c3d3545325ae0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 49, 3, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ae322bb10245d0fb3177051ab8b8b940(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dace80e590221230eedf7f0537464800(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 676], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_752cbf651c2a6801d6a15d5db2fc4293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 676], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e0f488da5133e810fe072828e66217dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 289], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d03f7364fd8708a717ac33469fd1ff1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 289], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fbb5ca82b76b30d6c051adec3ba23f0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d7f4cc822cbdf089651b404dc8346ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_526f81b2515ed71d19342a23c06ad49d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2a63191310eb798eceb633a6cc8f5c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_580c788923e7c93824ec8dbbbd6b971c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff3b4a1a7327cf7244031390082a7911
    def get_inputs(self):
        return [
            paddle.uniform([300, 256, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_13ced5f2adc86778c805fc7d133ac5f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f85b8e5a6b615c69f0d5f6c5ffe314e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 40], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_59b7624b249126f5532ab9073c24834b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7ecad22d0eb35d549560ce1142bfc3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 3, 8, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c4c9fa09f64885c38cae4564f5788369(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_70ed9c77a8af91c2950f758e4bfaf80d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_24c73a60377b81d3cf4a8b8be337227b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 72, 14, 25], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d513e5baa54eb694c0dc72dc4504cb04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 529], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c292fa5a9877aea9a4b280b0b4281dbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 529], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_580c788923e7c93824ec8dbbbd6b971c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff3b4a1a7327cf7244031390082a7911
    def get_inputs(self):
        return [
            paddle.uniform([300, 256, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_255647ec46429c20cada9ced99d81b80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3eb6b0f7fdb051472a73fa28acf2d655(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 3136], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eab1004a99a359232d541ba240d9504d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 7, 4, 7, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8a445d1738f7418465254f5deef69fe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 49, 3, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b98150dbb4608341e866819104fbe8ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7144693e26bf7d2aa93348250e6b3a2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d784fd88934fa68732f36fe9fef5ccef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_638cf165e69ab4fb2fbb0c9a848a6a6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 3136], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9cc31762d9f06f55e648cef0aee7fb4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 3136], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_571cf692323019369d62e7c06139221a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([6, 2, 1, 12, 24, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cfbf847b27522798385cb6ebf6ab3c09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4624], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8beaf87b484f256597151a7f6e4c000e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4624], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_72a546a1b133fc71ae1fc28b64c88a40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4624], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_70fbb745be89f999e9ad112c30138b45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ef34b82e95631a19f456cc5db06c42f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 3, 4, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_38d52f71bd32b0ac0d793cc5b9a2943f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_70d4d073b373705e937f621c2cbd5b0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_49fd8267114939086fa106335b4fdaa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1764], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9e551b3f03b7c5a496229918deb8ce80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1764], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_10378d470884fb2378b3f3630057006b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 5776], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0bc242f7ccca365277b500ebea599e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 5776], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_660a48b5bc8ad94400acfddaed898126(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 136, 160], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_84bad2e80e64c4aa29560d232c4b77af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 68, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4ef4371ce168214cbe70193146702c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 34, 40], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0751e13c125c02dff1c578355c3efec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 17, 20], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bb051979584b469475e7396fedfa97f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 9, 10], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4777cd160792f8a4ebb13e36d7dac10d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 136, 160], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_63dea711e0a9e56ee63fa1372fc534c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 68, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_345fd3f631a6873fd3b9760428dfd8da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 34, 40], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e35eb5b67a80ca532fa27c43f82dc2e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 17, 20], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_99288b87021250931e665b5b5c142019(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 9, 10], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1af3c256dfb472fc16e72a608839cea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1296], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_05d8aefdfeccdefa7793fbb1d956e7a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1296], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_17bf1bcbd19475dfca7e4c788acf58ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1296], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b9b3cec6b0a65fe2f92707852d0004f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b9b3cec6b0a65fe2f92707852d0004f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_09e4190d6f3e1d4a19f7609aa73b0ead(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2ebb97abbeca58f007fe77adf8d042c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 49, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_30be03993327f0539ec79e7a83825af7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1006b823370d28fe0b4e283c48eeec6b
    def get_inputs(self):
        return [
            paddle.uniform([8, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1b52d9f6aff3a6425593e7a80d78f7af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1006b823370d28fe0b4e283c48eeec6b
    def get_inputs(self):
        return [
            paddle.uniform([2401, 8], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2ed9767e648a4ddad768146129b43de3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 12, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2ed9767e648a4ddad768146129b43de3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 12, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_604a6d5727cffcf27cf6df687997a5ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f87b9772ebcb2e846a76caf49c85c9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([22, 12, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3656ef57abf16769cacb2df6a42083e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1006b823370d28fe0b4e283c48eeec6b
    def get_inputs(self):
        return [
            paddle.uniform([12, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2ac1ef85ba76017eae35b72d09e37f6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1006b823370d28fe0b4e283c48eeec6b
    def get_inputs(self):
        return [
            paddle.uniform([256, 12], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e9840a720ae3fa9362f555a63f631271(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 168, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7e33fb316ebefd1d0869d53a7c8f6fc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_68988ff68a1462b411fede2a53d40b16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fea1f4915b293f88b0af1ab2ee22f6bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1802ac0b999e60470ecfc0bdae171e2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e15b1c63ae768cad2208f88a705daa39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 168, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4db90acf49d5cded419d746a15abfeaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 84, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_341f2df5aadbecdd6d69dfb59ff8404a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 42, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d4219fcd766d75d035e75e98fec8fe13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 21, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d67386b5fd2d84b6a8881a917f45d736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 11, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_171c58020080c6dd29ea1bb5324c875e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 65536], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_414624e7f1fc38d55c6bec9897fc30d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ad9e2a5c03f70925c5a55778bd5534d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 784], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1b748b8bb0081310060602f8c355a383(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2bb74b5ad2ec6d42cef74ce6332c1a59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 8, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a1fecd33e5a0e54e23f9e4acdba31547(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_129187acba687eac4f2cb11d9043dbc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1707bb0c86535b5fcfcf3cdacbd388c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([576, 2, 96, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b1cc459ecb58c4d3651a1e04597cff7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([6, 96, 1, 1, 96, 48], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_98adf0bb1ca4d34613f67a8b1b74fb50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 176, 264], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_093eb555fd2d94a6adaeefe650440329(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 88, 132], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_20aa84d648a7c6e30a434b9f02a3f67e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 66], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a2fddaf3d6867a7c8aceafa56a93e265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 33], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e810b3510898bf79c63f3584fd776915(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8916a34a6f77246f4c736158589384e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 176, 264], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d6ad5be9b5aba04a696bfba41d133ab1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 88, 132], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aae49d5ccb5213449e35a1aa88b7d892(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 44, 66], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_90bf499a9986b105d299dd538cfe2b6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 22, 33], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2a3774f845781ee6e8b272d063beac43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 11, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_77b56dacc4aeadbc5dfb6837ece2fa18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 20, 128, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0846371a4a38e109427eb119ed2fac9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 40, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_af2a87221bf8ada66e4f3cdbb3fecbe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 80, 32, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b3b11348d9018e8e504c39c34cf0d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([528, 4, 96, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9ed02cf83a685e75d9e321520c20c941(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 24, 48, 2, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bf7e50e19e9a3cc7f5182195534b8340(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6a4fac9c6a27d252137dfbffdd8b3488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 116, 32, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_78cb9e8e3cea47f39d84f76067dbf829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 21760], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1fd06216078b967125ab7e775bf3e317(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_78cb9e8e3cea47f39d84f76067dbf829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 21760], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a44b5cc0fa9aa95c24d26737b8e8249a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 8, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ae2d99e4bc789693ded90f6bb5d9b130(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 8, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d7f4cc822cbdf089651b404dc8346ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_526f81b2515ed71d19342a23c06ad49d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2a63191310eb798eceb633a6cc8f5c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b763017fe5211f8ee35991c177c18e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 324], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7771c3130f76aa228a64d89087ba0c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 324], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3efffc63972e03a4914fc389471a3942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 324], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_22dcae9d2b9645c0ff426c41bbc78775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 3, 3, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bef1df5231a12cff2704608a21990399(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e249e557cd25e031bb67c8f6663a5ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0562ee5bebe73d8fd54d75900f25db9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b17b2ed7314cdac40106e3df9ee078bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d1fc90f02838e7d652812a54f55fbdda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 8, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bb0cfd4a715b8a3f33509738b814efd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 8, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7db72325e1821504e26fe70f623a880b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a58d3d79928d14fc4a2337984562a5d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d2f449b92ae925c756564d6958ee8b5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_233ecf34fed8c6c0c5f2df7996419f7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_826b18468967e22575874107f9216b53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f79f03d2f00fec7412adb81db919ccce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_66f31f8ea830cf05a131c5d75e08ccd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e714b5f06353f1294ea7269e166c9036(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dee6f62fea1e2f0743b9ba2f40945e79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([43, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fa73a92460ea79779c1da1cc12ee687a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3cf02a62af555b8d75726c467d10dae4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 100], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9cbb010207252647eb698a103ff83997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 5184], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_05b132fc4f0514190f7e6a5ccaba2adc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5184], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_da6db14b482c802350e0709f6c406e21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 5184], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9bab0169ac0a5ec08c4a84f1c2073c19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 196], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b4980cb50f1097461fc8aede16bbc938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 196], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ae95eaf7b6d4135496fb5811fc2cea57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b9a5a1e9733ca7b9e04ac18663eeac07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 32768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa049e690dfbbdeb996724ab6dd0924f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fbb5ca82b76b30d6c051adec3ba23f0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e33e3592d1920157826667a20b591b4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2a602033d845e512c401616d5b852bcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4240ca94d07b4c807a0cc6a9b5863e0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2c2b56113ebe9e235d2e3e20698bda03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c30c9d8ef71795b3802ec29856a4164e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2e41850a12aee8a1e9866d25082c65f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dcdb670e76440f2616e7b01c4ef290a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_39f2101bee3bf0235b515eb0bec83d76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([43, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8e1422c23fba5a7853bd8d7919032568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 7, 8, 7, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d0d406ac36859170856500365306f247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 49, 3, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_896b3adabf8bd09af17d053ca2012894(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2b9b3cec6b0a65fe2f92707852d0004f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_04105d88f2782c5dcef74c46602a774e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1006b823370d28fe0b4e283c48eeec6b
    def get_inputs(self):
        return [
            paddle.uniform([8, 196], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_63c16859c54e0918d1bb9d08fb9acbd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1006b823370d28fe0b4e283c48eeec6b
    def get_inputs(self):
        return [
            paddle.uniform([9604, 8], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bb368f9cb488510c00d1240b1303b42f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 196, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be9bba3a7c45a5985443ef817455cc2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a1dc269f36fc86ab2c0309e6c0c2d8b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 50], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_497d4b5925f437e5a09ce610ae6a8c5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([240, 4, 96, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_219c58bef434662afdc05fd8dbd60aa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 24, 48, 2, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dace80e590221230eedf7f0537464800(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 676], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_150afa69f7bc08959490d88c133ca265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 676], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ae1b7226d3e7d1cc069efb77bc80357c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef0231b550e32db08699d03359d0f80c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4ab719d653f7c802dc7d124e315fff04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4a799d760336cbc34ea001c59c84f9ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([12, 8, 288, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_571cf692323019369d62e7c06139221a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([6, 2, 1, 12, 24, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9396249f503e60fbda4d569205b0f190(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 100], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7fc216d49ba2f314fd2b2104d8fe3c3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 100], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3d2c7846b2dc0129bca34ee5ef9af7ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([22, 197, 2, 6, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0b914fddb1e31b1718251b44c629de1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 197, 6, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fe98e2707dd165a8c338013f459ee5a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([22, 6, 197, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d91dc18302ecafddc30cf88ecf8c04bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 200, 304], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d77c173bbdbf30e199fa6d8e9f552e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 100, 152], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_764e11159a1b19495af2259604ac8076(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 50, 76], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ff354f006feb93f334d33335c3d5a8b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 25, 38], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_de496da62840ce5d2a1d102afc0e8acb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 13, 19], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fbdeeeaf6ed2c0a17e2a7beb710bfef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 200, 304], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_187ff2daa94bbbd6165b6cf0715321b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 100, 152], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_911bd2cc27c4c8ebf34901038a1313fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 50, 76], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f503cbc3abba1b24abd812bcfe9eb0de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 25, 38], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b867522319f102301eab2781d01ce689(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 13, 19], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_98adf0bb1ca4d34613f67a8b1b74fb50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 176, 264], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_093eb555fd2d94a6adaeefe650440329(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 88, 132], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_20aa84d648a7c6e30a434b9f02a3f67e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 66], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a2fddaf3d6867a7c8aceafa56a93e265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 33], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1802ac0b999e60470ecfc0bdae171e2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8916a34a6f77246f4c736158589384e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 176, 264], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d6ad5be9b5aba04a696bfba41d133ab1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 88, 132], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aae49d5ccb5213449e35a1aa88b7d892(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 44, 66], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_90bf499a9986b105d299dd538cfe2b6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 22, 33], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d67386b5fd2d84b6a8881a917f45d736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 11, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0183f65568f6bbe835633df56852688a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_492ef3c25711446377733d7da6996c41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1006b823370d28fe0b4e283c48eeec6b
    def get_inputs(self):
        return [
            paddle.uniform([16, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3dde8557493435c5bd098d56e2249983(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1006b823370d28fe0b4e283c48eeec6b
    def get_inputs(self):
        return [
            paddle.uniform([784, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f408c9475eff46a986e8dfbcad32672b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 49, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d0971aad09db83711ddac16b7c1a131(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1600], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f5d83b464c527a3cb41604280935efbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1600], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ee60a5431a232695257d44e76466eca5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1600], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fd8fde1e7d64ad135ae840535b73b304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 144], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6c0cee8858633fd6a8942a1258e5f9b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 144], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aa08cec6952bffa11dbc819001b35b40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 36, 28, 50], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ba02d7e3b213a456812462f14d24bb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e06775daae16b01ea03bde3b1989d912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 784], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6f3276747125282acb3539b0e5f4ed8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 3, 12, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_23c7c469dc9e3c483b08ec877ad00ba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_07b13629af343b8c59c1b8a234840488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c449020fb9a7fc5ce4add9256cb05ca6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e0f488da5133e810fe072828e66217dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 289], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_44aad71eda2291e9dd98abd9565cf64a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 289], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bd46de9ef0884608ad7f1aa40241f80b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 289], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8449ca29c3692a1990656c55a0b5350c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 1600], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a652f0a00210cb98f838c262c8d1555c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1600], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e048f92e19e895347566d8b0bb015948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([6, 32, 144, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f46ee8e76c5ad96ce31bbd587e23a64c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([6, 1, 1, 12, 12, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_04c570d71360b2dfee2da242cff6b70c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88d10815f2af6d4d2305633ea16b28f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 3, 3, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c95ca2e9a9f658e3cde81778e6e5c656(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_76dcf1253cdf9128b3615d1686b89efa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be235555fa13931e6aa7d0394c258905(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 144, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be5a58fcf159eab9df56ae640693886e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 1, 12, 12, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_77b56dacc4aeadbc5dfb6837ece2fa18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 20, 128, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0846371a4a38e109427eb119ed2fac9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 40, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_af2a87221bf8ada66e4f3cdbb3fecbe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 80, 32, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_67e9a500a35b8d7ae4348cf4402e8536(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 160, 16, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f97aa6d260390695c1b331298ebe2296(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e772d5ab7a9f1aba295aef6b68d9316d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e041319f40fbd6ea5220a8f77b5a2278(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1b2363c8613e7c230b40ccb11729a151(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 441], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f35ab74d82baba884d6030ad15a21cc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 441], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e696a37a37f19aa1d5052c1b400ff95d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 8192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_204eda7b71824c85eb6b8f68f394c182(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 8192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_136e97252a28207283ebc7c176e337f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_feaaf475a000cb3a5c29f25f7ef81cbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_214effe12236b2021d3fe0660f935895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b41559a7ba5f9581f5466b9e04863e8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_255647ec46429c20cada9ced99d81b80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_76d6fc9fa1a654ce4c6e3a6611376915(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_42860ff90a7634a9015881c1e068e94d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c82e87b5def4c02b9fdbcbb082def0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([11, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_39dfdb46451847cbced36935331ebba8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1156], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1ca04425507204b90d278201aa85437f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1156], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bccc1a51e3b7b9d54eac0037185b1a0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1156], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_89b7f04228c0ce95e065ff2da62a55d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1444], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_463ea00d1d987966ef78858812fc9a3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1444], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7a13ef6a9b277ff2c396207a4466b9cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 3, 8, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5b8fbd4463678376cfee985864d01a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_08e7985ca32f84268f9bb57d0e35e06d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bbdc25bae2bed5beb1557fac7e956403(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d73d7e81dde011157fabf2ff0ed1b70b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_38d9ded5080b4ef60a41321f61c33550(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_70024431b5d3b028e67bb44cd017e6bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 128, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d7f4cc822cbdf089651b404dc8346ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_526f81b2515ed71d19342a23c06ad49d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2a63191310eb798eceb633a6cc8f5c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c5fdadd4f688e95ad026e6fe41901a5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 7, 1, 7, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_59cfdb2d9b01be3cc6db824fd7059845(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 49, 3, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b5adfbd7227e39c30f9ef37ffc014aab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d0aad9caddd2caf488cf3d4d43845b34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 3, 6, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e0ae0f4fad77039f4fcf67bd77613cd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cc2446fee0f5ff22be6eb17d799e4da3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cddf0c45f8fcc26cc588b20d98624b08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4e9b936813eedfa209d4e5f3dcca838c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 12, 12], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d7f4cc822cbdf089651b404dc8346ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_67a6c5f2246afa78b3604eb6ae97492c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 169], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4cc67ed4669c517ba8f915beeb7be9cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 169], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0802e7e5238d8c7d2034425253d5142(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 7, 1, 7, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9851ef260321949b51c3d3545325ae0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 49, 3, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ae322bb10245d0fb3177051ab8b8b940(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_af1551d8d99a50ed0981ab50bf4978b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 7, 2, 7, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fffec7177c247744e7c9d03fbf868273(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 49, 3, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5e7c9015cad1acd89bc8315e757b915f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f11c3aab51e0279e786ed29450c13f61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 784], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_960164e1896a2c232c63c1fd17e96e68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 784], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_69bbe167b312203bbce9a1da71bb44c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 192, 288], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7d2a57483b54e2a1d9c2e3be90761a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 96, 144], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a17053929d6de8ec2f293bea67d6af82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 72], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_335ee3d4188b9b9397bcb2342da6df95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_487471010425cabb4f22d0606dce1f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 18], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_56551d3c4a0a720ca7946fe7d7ae1e2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 192, 288], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4aa81f2814f3cca639f6f6027fec7095(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 96, 144], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_96d90a09bc645accb6476b2c1b34a49d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 48, 72], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_da0ad43db20f52d72269cbe0483c81dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 24, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_33a41fff19098d77bfa95ce88f38c6b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 12, 18], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_03523847f4cec53e4454326811eea086(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1297c88e0f24431ac8171c27cf3941b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([8, 8, 288, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a6c16fc3fec0e56c98e74f4ab0532961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 2, 24, 12, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2c2b56113ebe9e235d2e3e20698bda03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c30c9d8ef71795b3802ec29856a4164e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2e41850a12aee8a1e9866d25082c65f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dcdb670e76440f2616e7b01c4ef290a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_39f2101bee3bf0235b515eb0bec83d76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([43, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0cae6e34682e9f8b7b8b4411919f8cef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2704], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_322b009239647ef0f595ae952f3d4b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 2704], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c30c9d8ef71795b3802ec29856a4164e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_549f070539b7b40d3b41ebf30e1984df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 3136], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dc5ed6252c39229b7ccf651c3d79d36b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 3, 3, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aba22a9b5426b075d1aa79edb36b62ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bcf741682e50481bf54da1a07e30e301(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_43ce15e437f3e9a6dd95144ecd9c22a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 7056], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_df1875b86a247e2733a3077593ff850e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 7056], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e5fcf1cee535fe6d8b0c9720ce4f11cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 80, 144], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0cae6e34682e9f8b7b8b4411919f8cef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2704], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_35c0b2926add138e2b1661cdb1288ee5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2704], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8f23cfa6c930868228559308df202e57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_73acd15c3e78f51f4faee9e9ef708c5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 16, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3a4da0ff70523c43ad6662d607163054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b92a293c68589de46192dcba35a76839(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_150bcdd93be65f5f10be2d7d0fe2b6fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 7, 2, 7, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ccb4b9dd51619d0618eca60dc21cc8c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 49, 3, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d115320fa35c0fd07dc09241cb613c74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b17b2ed7314cdac40106e3df9ee078bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fb3df4c1e1cb841c4c3b9cca3af869be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6da2f0478b691041c8c6be4954f628ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dde2aceaafd00be29f1c65f7f9d000e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8464], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0fc61ece1bf25fcdb7b3f063c54ac4bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 8464], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_625cc2bc155b38908594713663c3ca84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([10, 197, 2, 6, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6f54bd68d5d9a29a643b7ef6148acfd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([10, 197, 6, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3dfea12418433ceebcc74b5e60929140(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([10, 6, 197, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7761986260a145b62db38968612ff015(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 3, 12, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58c8a70932a2502195ffb465aca41f2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2c9a7ed9dba3a761aab053f3554f4b2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_56187281a91855d28d568fbc575dd62b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 3136], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9cc31762d9f06f55e648cef0aee7fb4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 3136], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_31c02d7af5e105c8ce7e53b457a2e039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 7, 8, 7, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5cc43026d65da46739995e6c5a5c9e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 49, 3, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e7724832c3cb3806d2d54fb88638a702(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a8ba44956a59533a6b59bad4a06eb53b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 196], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6b11fa517732a4ffcadcc2bd8be7e91b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([20, 8, 288, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_09023009d040faf6f5e963b87e2c0ac4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 2, 24, 12, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_feeed630cb2f12e0335a0f0119d320a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b17b2ed7314cdac40106e3df9ee078bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fb3df4c1e1cb841c4c3b9cca3af869be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6da2f0478b691041c8c6be4954f628ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7ee8b88870654331d2cd6663d3ae8b8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([4, 32, 144, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_688d01c7cd7315f20d71d5c57e742037(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 1, 12, 12, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b17b2ed7314cdac40106e3df9ee078bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fb3df4c1e1cb841c4c3b9cca3af869be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6da2f0478b691041c8c6be4954f628ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7144693e26bf7d2aa93348250e6b3a2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d784fd88934fa68732f36fe9fef5ccef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2304], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8df4c61e024449d877dab8273fa918a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([4, 96, 9216], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e06775daae16b01ea03bde3b1989d912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 784], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d1b2cd29fbc3672acb222798bb8c4758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fb3df4c1e1cb841c4c3b9cca3af869be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e53dce3b12cb974ffb327ffeea30b71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4096], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_986e60762076a71b6f174b14c58b0351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a0ed618af7da2c0c4714048d60c9974b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 160], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_83cea391fe609237a5a6c9c520068fec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8e1422c23fba5a7853bd8d7919032568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 7, 8, 7, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d0d406ac36859170856500365306f247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 49, 3, 3, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_896b3adabf8bd09af17d053ca2012894(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 3, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4d7f4cc822cbdf089651b404dc8346ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_526f81b2515ed71d19342a23c06ad49d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2a63191310eb798eceb633a6cc8f5c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 1024], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cfbf847b27522798385cb6ebf6ab3c09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4624], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_de67d74c798e7c74a9b816ab976d9ca7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 4624], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a3de067e1d3cc82bdf1c61e55754f8f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e1561801c215fb2942746211489083f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff3b4a1a7327cf7244031390082a7911
    def get_inputs(self):
        return [
            paddle.uniform([100, 256, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_39dfdb46451847cbced36935331ebba8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1156], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2f3c26a95ba4f629f9523722006d0398(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1156], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8418d17457ff66582695cc3bdf457d29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6f2272544c7546b3fe302c4a61080568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 320], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54d86df0ce16c6a954ba512079ba85f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ba02d7e3b213a456812462f14d24bb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1968aea9614b59bbef5f1e3729129d2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_41d99ca11f0402148b261ebc6fed2356(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 6, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54369f8dac91673b29bf349fd34a6d5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b6232641e354a9a4e6f18303ce612cde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3eb6b0f7fdb051472a73fa28acf2d655(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 3136], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_59b7624b249126f5532ab9073c24834b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_629996cd8510630233d02a3566ad17f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_851dfd71bfc8db409a2e5f8968e947a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 361], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b4d0ef34c5633c319d2fa56ee44bfc96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 361], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dace80e590221230eedf7f0537464800(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 676], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_752cbf651c2a6801d6a15d5db2fc4293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 676], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_81e3d0f0a1a34267bae4f0467ad6d045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b2fdf1d719a40598042862b4951e32ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7ca3a642be252a9d2421682c5194c0a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_39c13d2318fc1c9326fa97fce3528c78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0d1b891af8a2f9eaad5d5a684fc2dfe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([11, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6ef34b82e95631a19f456cc5db06c42f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 3, 4, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_38d52f71bd32b0ac0d793cc5b9a2943f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_70d4d073b373705e937f621c2cbd5b0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b2f598c10084d89b36a7f21475db2cff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 6400], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e7d2842944f60b76b555ee77e3a28255(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 6400], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b6232641e354a9a4e6f18303ce612cde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a3de067e1d3cc82bdf1c61e55754f8f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0adbfd5483ce6c1d151b8f63589bb2ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 900], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2d40b03fa26c226a08f90b9edb2ae1f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 900], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c256f32f8aedc1227f3b4860264a0968(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([44, 8, 288, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bc7e318eb24b0810f3ff7f5526287366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 2, 24, 12, 192], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dc705b64cf6e4bcfe031c90a751357d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7b0ada04d0414ab28860a90af4ca5c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 8, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6fb78a80e9bf8ccf6823815d8abee588(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bb2b8c35fcff3846080f91aeb5e0e696(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_81e3d0f0a1a34267bae4f0467ad6d045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b2fdf1d719a40598042862b4951e32ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7ca3a642be252a9d2421682c5194c0a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 49], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_39c13d2318fc1c9326fa97fce3528c78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 12, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0d1b891af8a2f9eaad5d5a684fc2dfe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([11, 12, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a1c20a65d033032804dff68027406788(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 12, 12], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a9b063f68dc51be6775ab9ac44f11fb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 3, 4, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_432cfcba5900e1b1602d6d201fbe59de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_85ccb8abfc50f9aa8814b904f351d9ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_77b56dacc4aeadbc5dfb6837ece2fa18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 20, 128, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0846371a4a38e109427eb119ed2fac9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 40, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_77b1d460a9002c905d594e4c5ec5baf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 3, 6, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c745dcc6a501a01553a0d4fcea68ad1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fefaffc49a69262e77361a876e3bac60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3066931b0d4e2829f9629fbaf746cec0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 400], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_82b5f4ba6e092b4202345bf56a6dd66d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 400], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f0e8a339aded4194ace1c4e9b7e1bd6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 400], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f71c2c77741e9a1597104872764b8b30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 225], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_832eeb14ece0b09eaea5ce4f73020668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 225], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_66c0c7f68786328d29bedffdd098ed40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8ba2ae5d6c77506c4370c262ba519d02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 19], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_67a6c5f2246afa78b3604eb6ae97492c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 169], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_066964132c054e9579ac9ea2904a43e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 169], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e605695aa7718907155c20bce8b70f72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ae1b7226d3e7d1cc069efb77bc80357c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef0231b550e32db08699d03359d0f80c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4ab719d653f7c802dc7d124e315fff04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f633453fbd446be509208cb7387aef92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1960, 16, 2, 4, 6], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8e0df7e277edd2e2d3e1e385c92292b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1960, 16, 4, 6], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d558abe8619f611d71033c79b84be612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1960, 4, 16, 6], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bbdc25bae2bed5beb1557fac7e956403(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3237f091273d4416141ddd70710f2a3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 576], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9ed7dd6fc6bf25e09b6a99694304941b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 200, 272], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d5a20e798d9bde39e8e696fddf441699(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 100, 136], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_87b56b4788a0b915e6e996166b848fb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 50, 68], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a10d6d19792c7fb573483496fbe0bf8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 25, 34], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_af36a824fca0cb5c24a3b6c57c6afd7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 13, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_09d1ba98da4942055ac6504f90805d62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 200, 272], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c63ba5e19c3db22dd6142ef20a003643(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 100, 136], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_19f9d8fbc4a92f5d19b401fcf6bc019a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 50, 68], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8ef8733ccf90324dd716e003354de7e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 25, 34], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8067eba4d1d28f1d2c03cc26be70a36e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 13, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bc07697d818717e644d0e7181cbc1125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 400], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e79aef97fe0a6a33f825742a1a77fa9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 400], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fbb5ca82b76b30d6c051adec3ba23f0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e33e3592d1920157826667a20b591b4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2a602033d845e512c401616d5b852bcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_67fb4fc916ac43b92555887a7240f510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 25], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c5fdadd4f688e95ad026e6fe41901a5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 7, 1, 7, 768], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_59cfdb2d9b01be3cc6db824fd7059845(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 49, 3, 24, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b5adfbd7227e39c30f9ef37ffc014aab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 24, 49, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9de9356c6a9d32824bb1c0ca433ecd8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([2112, 2, 96, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_060247a443061f85f32dd510eae2bf78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 1, 1, 96, 48], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffc75680a3eecca74c0b08f7f211f723(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 784], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_960164e1896a2c232c63c1fd17e96e68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 784], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b5bd1e5bd99c62029937d43a0ec3a8b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([96, 4, 96, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4800277c5b4456fcad158548a10f10a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 24, 48, 2, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0f1d1202df6f5b5861813479ecc7729e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 58, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fbb5ca82b76b30d6c051adec3ba23f0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e33e3592d1920157826667a20b591b4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16384], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2a602033d845e512c401616d5b852bcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 16384], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()