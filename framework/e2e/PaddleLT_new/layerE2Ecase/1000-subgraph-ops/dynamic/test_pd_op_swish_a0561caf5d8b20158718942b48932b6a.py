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



class PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_017431d15a8b191b6eeb9546820d3dd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_658839dae9444044813b3dedb733118b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_66bb52151523a843b014028e8b6236f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cadb48308e7f30f8ad4ec8960f126840(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9706c0668980eb917e55ccca1ea960f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c22a46a7498aa11a3c7a06f7de9dd8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_017431d15a8b191b6eeb9546820d3dd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_658839dae9444044813b3dedb733118b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d4f754d5fc46f0b0f770a4373a19ec12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4618630d47ddf30941128ecd8d32dcb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_55f239a6789b77324a2dfb2aca7bdefe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_658839dae9444044813b3dedb733118b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_854cb0bb28742efe9cc375fb91aa3c0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_795eb4a067b9cee02faa8f5951c1c614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_53352624fb2b4141d4ce53b9d11ac466(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_55f239a6789b77324a2dfb2aca7bdefe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_658839dae9444044813b3dedb733118b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4618630d47ddf30941128ecd8d32dcb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4bf6ce53ebc7bf536b309d1e0f271a71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_53352624fb2b4141d4ce53b9d11ac466(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_89f75cf8c4cf2d57df175ffdc6d8086b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65df3ba9fbf207f0f9e30ef1f9d6c0ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9706c0668980eb917e55ccca1ea960f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c22a46a7498aa11a3c7a06f7de9dd8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_de09b22962347d76c953a5f871bd4937(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4114254c0a6eea3a9f2635031b715039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c2b206d7c53627aa0a0fdef286adf70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7b43bca93fb23a06af92dc7ff1d8d72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffe262486f8a3b5ffb7a1d73cabfa7b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1b1351e1eb8225bb8ec0728caf4ad683(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7dd2f14125c71912031f11ab6047ebac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e386e49161eb0a8a25f322c43591054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1ee347b38684fa76a2548a71c2954f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_47fbd020c4f214436d5e456cf81884d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffe262486f8a3b5ffb7a1d73cabfa7b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1b1351e1eb8225bb8ec0728caf4ad683(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_60ea0406daf12be15195b3c14f76086d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 7, 7], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_64872330be930a614788334caf21ec02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ecc7aae00fe6d9893548f7bd275ee7a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_854cb0bb28742efe9cc375fb91aa3c0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4341ec61869a1b6669791d2c37118ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a3bd3b02aba0dc4b4289641ada9a2b48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_89f75cf8c4cf2d57df175ffdc6d8086b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_65df3ba9fbf207f0f9e30ef1f9d6c0ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4618630d47ddf30941128ecd8d32dcb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffed5973c6e029e90413e9df5cd2e978(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c22a46a7498aa11a3c7a06f7de9dd8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1ee347b38684fa76a2548a71c2954f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_47fbd020c4f214436d5e456cf81884d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_53352624fb2b4141d4ce53b9d11ac466(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_64872330be930a614788334caf21ec02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dd6d88308b025281a8f744cfc7446c2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_47fbd020c4f214436d5e456cf81884d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_53352624fb2b4141d4ce53b9d11ac466(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4114254c0a6eea3a9f2635031b715039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d2dd64c8345a6b6bf65c4b15a561a3dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4618630d47ddf30941128ecd8d32dcb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0c2b206d7c53627aa0a0fdef286adf70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 112, 112], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d7b43bca93fb23a06af92dc7ff1d8d72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_de09b22962347d76c953a5f871bd4937(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4114254c0a6eea3a9f2635031b715039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a4341ec61869a1b6669791d2c37118ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a3bd3b02aba0dc4b4289641ada9a2b48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 20, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_66bb52151523a843b014028e8b6236f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cadb48308e7f30f8ad4ec8960f126840(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3283e61d601111c9b800c6fef4ecc065(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4114254c0a6eea3a9f2635031b715039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dd6d88308b025281a8f744cfc7446c2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_47fbd020c4f214436d5e456cf81884d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7dd2f14125c71912031f11ab6047ebac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0e386e49161eb0a8a25f322c43591054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ffed5973c6e029e90413e9df5cd2e978(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c22a46a7498aa11a3c7a06f7de9dd8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b5a6f03efb0f0d41fee37c445acbcd
    def get_inputs(self):
        return [
            paddle.uniform([43, 10, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()