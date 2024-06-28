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



class PrimitiveOp_7c26dd0482ce61cb855d681e00824b22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3f69ae6a22c2c02492904cf63db701c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c51bb44b612b667214723e983c45a7d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1, input_2):
        return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9901ba6fe826c1509ad6ccad0de75a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_953a844ef94ce038bbd9accef9d09368(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d093cbafb5ecf833e9759cee01cecf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35546d37c33b13695ce75721813c7a60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_628becc8b8e684e6856d871ca8498c4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([16, 1024, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1414d6eac63df657500ee13e379b4cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09a05cd35982e4dfc0ae607d892ee913(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_31edaefa77dc647c90c0f0c92963a274(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([128, 32, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_128c4958186988b76af12d34660171d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7eb49552ca657b6ea4a59213acc6c98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edb6a20db4cbaeac95357fe5785ac1a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a4c146996e781d196ad4dfe1f5d0575(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a15dbe0682627b69c6300af1ccb0b00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65e8ad3109beb0527afc084172c1d176(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_77ae237176d3448ebd0710904ba2b87f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c1619bd1e4f0d2fa8c44939dceb78545(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27425f52c82734e2937c33c2734c4508(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_126eb8d72429b0fa1bd153d77bc057c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0b26ee5ce3e4651c2da9ee9286422ee9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([64, 256, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_94155547f3f37867796404cf9c3a158f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09a05cd35982e4dfc0ae607d892ee913(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70f15b3595394c601d3aaf4008a73055(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7eb49552ca657b6ea4a59213acc6c98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65deb7ea78e281c8b862dbd09cdfcc6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c1cb17d736f203c30f9570cd30128c17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([16, 512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90d209b1d1006fd9800d4b74e34b508b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_800319d0717b0f1f761e5c68fa190479(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5442c5c42e801a652951c536332117a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([8, 512, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_128c4958186988b76af12d34660171d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ca969ae3ab6fc18f677d1a2d54dcbd58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8b5e694f7201c2dc75aad0949e0b133(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_118cb12e8767536d005885488c8799ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_744896279ffe9e6fef971cd6cc8d62b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2b1a82c4dbe487cade56a21b9a23b870(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2b1a82c4dbe487cade56a21b9a23b870(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ffb1e5df9118978308b392880d7ca33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2fff17f6288c456669e2cb467d574fef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3fe2beda15b0f57ebe9830c0bfb83f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86a458ccc756d0713f3a0be1eaa13b71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_065da33b32be5ec5806c1ce4dd51aee4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9901ba6fe826c1509ad6ccad0de75a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7a0153608d212d49b6f8b4e510710ea4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90d209b1d1006fd9800d4b74e34b508b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ca2db714bd92f4e4d47ec71ee95b0ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ca2db714bd92f4e4d47ec71ee95b0ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_00ce5ee9cb262780b65b929909730f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90d209b1d1006fd9800d4b74e34b508b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db57ddb7ab532ba6f6b737f177fa5cd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27425f52c82734e2937c33c2734c4508(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf5d5824f2aa58e1abb325032b5fde3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_41be9e90fc13077c16a90c8757251e84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dba8e486fba8ce1c3ffd7d867b32bab7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd08e06e30896c7329664451802dfd05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6291fb2187625f87d2a453c3e03e94f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64d4379ba4f35ee1f864587b90362ab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([16, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3308f2e9258b7c2fc7bbc0ee3ff99c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c3f69ae6a22c2c02492904cf63db701c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7990ee7753f9dc72a529c1654bf2b0ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b708f2b0824702fca0e0f5f6b19d478(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_126eb8d72429b0fa1bd153d77bc057c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_33fc1f4d4e18cc3bdae913228125ef5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0b9e31773ef44b427f0147d4dc730779(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4a3acc994e1a58cdc9d283ef5840034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ffb1e5df9118978308b392880d7ca33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06ed4afea1448b8ece94de27418a1f41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1c36c0a1793ecab7bffbb0d7e432607(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([8, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_369e6436e879adcd4b7fc6ec46c0115f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_744896279ffe9e6fef971cd6cc8d62b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f62ae4bc5fa91590e12d66f96f69279d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edb6a20db4cbaeac95357fe5785ac1a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65deb7ea78e281c8b862dbd09cdfcc6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4610eedc9013a53590f1f5d6b3fcdb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0050a23b175841492f1eeae87a5e0b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_690123226950bd11f0389b2109a59650(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ffb1e5df9118978308b392880d7ca33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0ca73e3ece762f82a4747576baf1406(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c2c0ef8ff5f3754035977c70637cbabc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ca2db714bd92f4e4d47ec71ee95b0ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a4c146996e781d196ad4dfe1f5d0575(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_59c6f082ca7cf3fada800e6285824d50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_22c59abfd5abb1aa778f3f8aa956c086(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c9b39f957f64d66f971be4d8a4a84b82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d56a51de272c2eb613e29a1a9fcf3be8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_369e6436e879adcd4b7fc6ec46c0115f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27425f52c82734e2937c33c2734c4508(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d56a51de272c2eb613e29a1a9fcf3be8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_354cb4864deb5a70962fbfbf1aecb0c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_128c4958186988b76af12d34660171d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f62ae4bc5fa91590e12d66f96f69279d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27425f52c82734e2937c33c2734c4508(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d52f0c36d9049761b21544b6f2febbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_128c4958186988b76af12d34660171d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3fe2beda15b0f57ebe9830c0bfb83f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ffb1e5df9118978308b392880d7ca33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_42bbb138d952a2deb0b52e2d261f8992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d3edc0d067ca6ba1889cd2c2d51b411(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06ed4afea1448b8ece94de27418a1f41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_baa1d4d5152818fb605fb3eb45c91df8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1414d6eac63df657500ee13e379b4cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96528b98e8e02e096bea1e54f62b3ba7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_94155547f3f37867796404cf9c3a158f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_84ef6a6ae503ffe194eb34fc63ae0b5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0d093cbafb5ecf833e9759cee01cecf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_369e6436e879adcd4b7fc6ec46c0115f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_065da33b32be5ec5806c1ce4dd51aee4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_94155547f3f37867796404cf9c3a158f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90d209b1d1006fd9800d4b74e34b508b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d972a88105188504492e1538fdba9e59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_800319d0717b0f1f761e5c68fa190479(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc42a2685d31d42eca2c132f5cbbfc07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65e8ad3109beb0527afc084172c1d176(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4610eedc9013a53590f1f5d6b3fcdb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_42bbb138d952a2deb0b52e2d261f8992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79f90b80ad5a2f93c8656e3d317a8900(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_49b19782d677ff7651b95c87d9ce0774(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6291fb2187625f87d2a453c3e03e94f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c123ac0a520b3b1c0ab3b9cee244714f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_22c59abfd5abb1aa778f3f8aa956c086(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e669ad09c380df3f9a7301853038adc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([4, 128, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_369e6436e879adcd4b7fc6ec46c0115f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd7967a475d9da6aa2f212e44123921a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ae9a8c4168bcaf2bb29a1f4600f1422a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_41be9e90fc13077c16a90c8757251e84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7420b05b4b19753aaf19cc744686e929(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([4, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dcd1960baa7c0f3f1ff0f0fe111587cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35546d37c33b13695ce75721813c7a60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc42a2685d31d42eca2c132f5cbbfc07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f62ae4bc5fa91590e12d66f96f69279d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f62ae4bc5fa91590e12d66f96f69279d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c26dd0482ce61cb855d681e00824b22
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f113a47353963cba52f4af0617acad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51bb44b612b667214723e983c45a7d9
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()