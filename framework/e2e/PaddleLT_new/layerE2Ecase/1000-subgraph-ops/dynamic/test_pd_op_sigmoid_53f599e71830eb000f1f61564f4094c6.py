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


class TestPrimitiveOp_c3bd8a4f68207021b3d602e2f93dfbdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 60, 60], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f085540995bd2e64e6789a49d3a9b46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54552b6c6deada766906bcfbcb884666(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_accbc209d1779c5a33822abda39eafe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7355f4f86f64140a85de57b5393a4e06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 20, 20], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c2845c3893d95e4cf168af342f5a946a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_836bacd50c30aa7d4757ec3e3655db04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
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


class TestPrimitiveOp_d8e5218f181bfcfbf280304ee6aa407d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
    def get_inputs(self):
        return [
            paddle.uniform([15200, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e0dc570e50942d8021d6aa4d46257cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6dece2c9f5f2b1387696a810febc209a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_29902c8a9fe8608ad66215e47e8b7290(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5503e0b835017c3b258e2cabd3d29055(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 46, 46], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cff31fdfbc6a00e4bc65734996d0c686(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f085540995bd2e64e6789a49d3a9b46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9c07abc9411a9da82a5a624781516215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 80, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b8394fee2f0f45efc9f3404436d5ec6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 80, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_66698b4e426b7fb72d720e795edad5a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f085540995bd2e64e6789a49d3a9b46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_efebca2ab3fbfe545f3e051fca003e70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c69196375acfb481ae7e52dabacc3bc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e038233148a4532e503c16fbde78f29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 40, 40], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bef5474447142ec159f2cc293212d5e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 50, 76], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_251d5ca82ba9aa4bf2c9205d0a2bed51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1f085540995bd2e64e6789a49d3a9b46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_66698b4e426b7fb72d720e795edad5a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c69196375acfb481ae7e52dabacc3bc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3369c67859fdd80817aa280a5ab39edb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 100, 152], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_02ebcb557d55e493207e78b39923a21a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58b5d1ea11550e111c735dc4b64872cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e5914db4468059943f9dcdcd21ae09f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 13, 13], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b4e0a3b55fcd71e4ba67866bbfde7363(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 120, 200], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2730e0e7dcefc57ac50046541055e9a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2deb5592529e59b9b4b48ce98f8ef121(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a825be05262760e71cc09903868b5fc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8250176079eef2646513da7720742b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 48, 48], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_98c071326de96a645455659d864576c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 48], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2730e0e7dcefc57ac50046541055e9a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4fad518fdebbb76505d9f5a267aa7b61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 96, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e89d053a9ddf602d946b70b1f830f412(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 96], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fbf66e6ddf3e5afb6d25f05be45a6bc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 7, 10], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e0dc570e50942d8021d6aa4d46257cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_038c5dc2cb935758d18c2f6bd0721f84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2deb5592529e59b9b4b48ce98f8ef121(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
    def get_inputs(self):
        return [
            paddle.uniform([3800, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_936bca484e44f4a09d6a245134b8d7e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b205a7d734c8ad5a7bb4a5654359e6d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd69b90238e5681e3c7a86c236094023(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e6098ab2f59654becd97bee69916f119(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f89c7b3e5d8f09c00d02b6d282edb13e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 17, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cff31fdfbc6a00e4bc65734996d0c686(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2d74127219b72460f424f2fa2ed97801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bd6bf8c8d1404bd642e8394a3162431e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
    def get_inputs(self):
        return [
            paddle.uniform([2204, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ca0f02265b2d3c57b56a0dc1bd2ab6a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_06c2ec551063e6ae69cf06d167311e90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_39364f145b447e79614934c4ae8f9944(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_26992522ae3c81ef5100efb83931caba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 23, 23], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8250176079eef2646513da7720742b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 48, 48], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4b22ea9884d3ef2350c11a52419b830c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f8a7e9a4b165650f2684932105f2ab9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_495e698961e7517ab1bdc986e790012d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 68, 68], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_adf0f6df0bd4f24e855d3c37a1ccb0ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 68, 68], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_bd3cced97d65392629da379b7700fdb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 112, 160], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ec4e106554cd06f85605b1ae76b8ce5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_caffbc48e9fbbb74e5ad8efc3242d941(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 42, 42], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58ad366db27d3929f1edc80fe7f3eba7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 76, 76], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be892585b401cb5074881a1538d99330(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9f99cab6b475ba79920460755433366a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
    def get_inputs(self):
        return [
            paddle.uniform([950, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b205a7d734c8ad5a7bb4a5654359e6d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 1152, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f1dbf653b61db43948e804bbae5f22e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c9c700042784b98b1a8cc1ae0a38646b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_251d5ca82ba9aa4bf2c9205d0a2bed51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2730e0e7dcefc57ac50046541055e9a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00a7bd3451d2ee4702dc6a6fe56b7c3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_39364f145b447e79614934c4ae8f9944(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_16009431b78c7506bb8a30e84727b142(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_aff2209516a9759380ddf28d03f05187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.10129356384277344]], [[0.2721717357635498]], [[0.3421568274497986]], [[0.2625434398651123]], [[-0.31282877922058105]], [[-0.21825647354125977]], [[0.684687077999115]], [[0.47957825660705566]], [[0.03648719936609268]], [[0.043250858783721924]], [[0.4967327117919922]], [[0.07854483276605606]], [[0.06093655526638031]], [[0.8404895663261414]], [[0.4349615275859833]], [[0.8172653317451477]], [[0.49775460362434387]], [[0.2955150902271271]], [[-0.32571521401405334]], [[-0.2340446412563324]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_e6708534d759135f7c55343b47096b6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e386a186da4c4856769e6502eee155f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_efebca2ab3fbfe545f3e051fca003e70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a27e6047fb3f706ad51506ebaa2b068c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2d74127219b72460f424f2fa2ed97801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_91990971eed6d630eece3ebeaecfad04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 18, 18], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_81e7910f21f52fc6d5bd3f6bbed2b0e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 18], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54552b6c6deada766906bcfbcb884666(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88559b723eff78e66d74a67fc0ac99c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9db3d1b277ad352e0ac5564b9123d78e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 8, 8], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b3807c1a155334d558c4b39f6d1717da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8e26df0605e12e7e5328bc67e2fc0ac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
    def get_inputs(self):
        return [
            paddle.uniform([8816, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d2d674fb46eb5f1510319cffd8d40154(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 72, 72], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_04dcf3ce876ad9929accdadfff4b1200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 72, 72], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8d1a5999ed00b7ebbfe1ff8ed502b44b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 14, 14], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_06283396c702176ab57a3b6fc8ec9131(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 15, 25], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cff31fdfbc6a00e4bc65734996d0c686(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a825be05262760e71cc09903868b5fc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fa0a47486bfbe495296f83e516c329c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_00a7bd3451d2ee4702dc6a6fe56b7c3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e6098ab2f59654becd97bee69916f119(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ebbb7c186e12511ff6c0a90a1f0b4625(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 10, 10], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_995ea5b9bcd4440c2de67234bcde1e16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a2315b42eb87f21fa7298212ffa61edb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_caec4210726164c1e86ebe2fbdd52fe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 56, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_accbc209d1779c5a33822abda39eafe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e038233148a4532e503c16fbde78f29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 40, 40], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0609b7a99eac3e16d714ebc6dd31409f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 40, 40], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0db06357aee0e29f592d25f7f683658c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 12, 12], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b9280f6771b869e16ff627dcde523d22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
    def get_inputs(self):
        return [
            paddle.uniform([150, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f89c7b3e5d8f09c00d02b6d282edb13e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 17, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_95f5f94abea5e4e1ad730726c34c00b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 17, 17], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_66698b4e426b7fb72d720e795edad5a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e0dc570e50942d8021d6aa4d46257cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ebd400e615d3ec40b9718bb389d9b2f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 40, 40], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be892585b401cb5074881a1538d99330(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a18e5f08b535002fdf2588e4963fac92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 192, 288], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c69196375acfb481ae7e52dabacc3bc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_19f23fd582407c8aced65a425146ad37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.1703701913356781]], [[-0.20227152109146118]], [[0.20821166038513184]], [[0.0738363265991211]], [[0.47628629207611084]], [[0.1800503134727478]], [[0.19318795204162598]], [[-0.350565105676651]], [[-0.19503796100616455]], [[-0.29024505615234375]], [[-0.30695605278015137]], [[-0.4363560974597931]], [[-0.47536394000053406]], [[0.34204936027526855]], [[0.25050705671310425]], [[0.3783673048019409]], [[-0.3022617697715759]], [[-0.055722594261169434]], [[0.045115113258361816]], [[0.0799904465675354]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_e6708534d759135f7c55343b47096b6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e386a186da4c4856769e6502eee155f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ad7affff01ec63e92fc7da6af9f4dae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_833d9191da987059c135d74d013d27bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
    def get_inputs(self):
        return [
            paddle.uniform([70, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eefd01895dec0c43ea62c4e6ee703c41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_16009431b78c7506bb8a30e84727b142(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 1152, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be892585b401cb5074881a1538d99330(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_dea5249f74a26606e3ff81642e3b245b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7b5204eec0d62bade289bacfca54e1a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_26b85ddeff5e30e969efdaaedeb40af9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 21, 21], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c69196375acfb481ae7e52dabacc3bc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 672, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_918735853a1c80a7730d345ec214526c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 34, 34], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e8e27ba7626a6639c7b3f4254f30c2d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 34, 34], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3c3ab03f6a1044548ac85765d9db698a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 7, 10], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_44a3fac3807f14ee157d97e9776bf2e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 14, 20], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_f521f71544218135a8d1bb43eed2efb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 38, 38], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58b5d1ea11550e111c735dc4b64872cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_51cb5661c39a1ff61fa481f1e8cb4b8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3f3815e0950bc8b883aa7a0ed6c85849(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_038c5dc2cb935758d18c2f6bd0721f84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2d74127219b72460f424f2fa2ed97801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_66698b4e426b7fb72d720e795edad5a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2d74127219b72460f424f2fa2ed97801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e5914db4468059943f9dcdcd21ae09f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 13, 13], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_760452c83cc4effc60a5943e155fcd34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_936bca484e44f4a09d6a245134b8d7e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd69b90238e5681e3c7a86c236094023(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_052a3844fdb6847e37a3f532da7d61ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_038c5dc2cb935758d18c2f6bd0721f84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_459535f7865a682a31d302b216a7d1ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 13, 19], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7f4535f50a25c507a0777fe909364efb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_105a2edb25fc7bb55d818881d73c3f31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 84, 84], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eefd01895dec0c43ea62c4e6ee703c41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6a8fcd21d3fbdda311e0ef8f3bc59df5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cd69b90238e5681e3c7a86c236094023(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_995ea5b9bcd4440c2de67234bcde1e16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54552b6c6deada766906bcfbcb884666(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b3807c1a155334d558c4b39f6d1717da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_fcaf0e5f66e69d3679f03718e1d318c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 92, 92], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_88559b723eff78e66d74a67fc0ac99c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 480, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b21de21f77d183cb98a461ea9f7c3223(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 56, 56], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_836bacd50c30aa7d4757ec3e3655db04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54552b6c6deada766906bcfbcb884666(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_54552b6c6deada766906bcfbcb884666(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_babfc7e0ffd0ba334e80f8ba7ebcabb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 25, 38], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8250176079eef2646513da7720742b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 48, 48], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_038c5dc2cb935758d18c2f6bd0721f84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([11, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_be892585b401cb5074881a1538d99330(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 144, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_c109c0884c9c2762ed5f216692bfe757(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_412a31db78afd75b78436c40538933df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2d74127219b72460f424f2fa2ed97801(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 32, 32], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_495e698961e7517ab1bdc986e790012d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 68, 68], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_06c2ec551063e6ae69cf06d167311e90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_8376953cddba8387ab9940eb724a81a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 120, 200], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_918735853a1c80a7730d345ec214526c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 34, 34], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e018fe22000ba91edc217ada35d95b19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 24, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_251d5ca82ba9aa4bf2c9205d0a2bed51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_3c7f8ff7925dd1f84dc2e7a97f0af2da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
    def get_inputs(self):
        return [
            paddle.uniform([551, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_29bee934b1b52d51e3faf0223cf8b4ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 19, 19], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e6098ab2f59654becd97bee69916f119(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e8ae3fbfdcc235e439ac9d3a20d944d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 28, 40], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9c07abc9411a9da82a5a624781516215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 80, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ec4e106554cd06f85605b1ae76b8ce5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_32efb8aa1a044aded6b822b9c52f50ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_4b22ea9884d3ef2350c11a52419b830c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1490f0f7095d823aeb625902691a4938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.25554007291793823]], [[-0.08294905722141266]], [[-0.45818787813186646]], [[0.44687125086784363]], [[-0.3170972168445587]], [[-0.037483394145965576]], [[-0.16191241145133972]], [[0.09497104585170746]], [[0.664016604423523]], [[0.599393904209137]], [[-0.19525286555290222]], [[-0.4981592297554016]], [[-0.2611096501350403]], [[0.2058744877576828]], [[-0.023603886365890503]], [[0.03177797794342041]], [[-0.06511509418487549]], [[-0.4945560097694397]], [[-0.018913060426712036]], [[0.45039406418800354]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


class TestPrimitiveOp_e6708534d759135f7c55343b47096b6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_01435a9ec4444f61b2d93bd0afc9ebcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_7355f4f86f64140a85de57b5393a4e06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 20, 20], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_0247f9c174fb44b79dbc49ca2b3d20b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 20, 20], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1b2e522f77643f4ec1e399a3e9ed06a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 15, 15], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a2315b42eb87f21fa7298212ffa61edb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_a2315b42eb87f21fa7298212ffa61edb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e5914db4468059943f9dcdcd21ae09f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 13, 13], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_accbc209d1779c5a33822abda39eafe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_58b5d1ea11550e111c735dc4b64872cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_eefd01895dec0c43ea62c4e6ee703c41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_6e0dc570e50942d8021d6aa4d46257cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([43, 240, 1, 1], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_05ef2e6b5024f79b21e6188c3e4bb1a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 20, 20], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cff31fdfbc6a00e4bc65734996d0c686(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d39baeea5f90e173dcf6ffba8523824b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 192, 288], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_2e2ec5280f4392699e1dfc1bb4c92d3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53aca09af1bf1031591a5d021a60c829
    def get_inputs(self):
        return [
            paddle.uniform([247, 80], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_06aff06bf3f766b6a74e56b9681fcbfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 28, 28], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cff31fdfbc6a00e4bc65734996d0c686(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cae12e17801d9cc2b63cadb6eed2e1d
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 128, 128], dtype='float32', min=-0.5, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()