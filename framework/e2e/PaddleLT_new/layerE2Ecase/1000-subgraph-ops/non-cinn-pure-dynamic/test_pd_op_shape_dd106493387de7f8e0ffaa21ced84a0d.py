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



class PrimitiveOp_54be9492f0763127dca1a4c6d782b47c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d557a16f2a924d850806deb5910228a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_831eb409930319945a13e30d8d5c79b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a41f5b5360a1e32500abecf7985915b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([43, 112, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_71021bd69d7fd83acd87dbf59891afdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([16, 128, 16, 64], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.shape(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9c4082122881bbf36952c24703e8597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6773e21b79b9b9f1629bacacc30d3ed1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7c6b5afd383de1048b62a9fd357b8275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_88e566da1a4852f1acb96dbc8fefd5be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c98036a66c26af6be178d749ed197c7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08d78073bd0d59c07cafe3d49499609c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7f19820ac0fe7b92de11363fcbfec35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ea2d037f5b7e9e6c6b08cee93f0a9b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([128, 320, 8, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef14067aedc8d9c44ed39bab6d9b6f6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3cb6f8ac4c0ae0e48a4518ca0a7d82f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([43, 7, 7, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3913309bd95f8bd66a7e830e3bac23b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0d76cce2fd3b6b2f535a49421eaac24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ea24720838f55f0a8c2b9071d934243f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([43, 80, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f42a2ff0f0df1d1b416203c23217cdb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1175e9673c1ee84c08c0aa5f91ea7fa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([528, 4, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b34ee9619bd942a304daed63d2abc518(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d508f7fada55d70ffcfc1b681175c77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([12, 288, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e62cb972d795f95f179fc17b2d6abef4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([8, 320, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a401dfc919cee88bca8b3b71d3ed096c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc50d4aa19470cb5af451761249be5d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([8, 160, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07923ad1abbb797e7f79a05f47e93d06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3fcbb27cf1a6ca832548fbfd03eea2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_22a0832aa871961e19cde036f3383213(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0657fd675bd1da58079574da8fe3f4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 32, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e6c92e00d6a93507c807d39a6931ed9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([384, 2, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ecc3cf8a071b4ac469298293d1d5d39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e73ea87a5da733cf18e7d9023c29a895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 28, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f23ad82df9c252bdd7f7b7064587d218(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([16, 128, 16, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd0ba775db6ee6f750ce80c7b34d0770(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98530ae7fe7c05a79347dc63c5c5b1d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_490e70dad1a5d186e717bac0e699c320(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38f6c0044a2cb834c3eca4fe714ffef5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_177eb009e25462c238905c43df3c35f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b7746e46b086ba3419f6ca0e7f47ea9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f12f15a5ad1eafc7950c307f425effc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3687ccb95800af5cdd99c8badfd889c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([8, 160, 16, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5258bc07ea2e66ece6950f39c698dce5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0d76cce2fd3b6b2f535a49421eaac24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ea1350bfbf346fff2eadbe0774891bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9a49564a7562cfd0671375cf9f777a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_696b1f769354fe0699eaa41f13a97c9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da0617ae02509e1a17ba27c97d10c0cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_94368a862357a41385c02e136d220ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e2c79564b36878d6096d35d7751e2b1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3900645914a86542312805b5a62da657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([20, 8, 288, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b456ff49170eefcf1357174a87c7bb0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e2652fd30caa3cded7321b75a5ffe38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3716240e64e65b680aeccf8cd703e38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([11, 7, 7, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc210ba8bd30cc5058841cbc571d7bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 1024, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a495de9c3bff44810cb92520fe884db9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 256, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_568625526b5542446bb997a73b544ed8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([576, 2, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_351542fd077d627a8c8292110f08032b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 144, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ce3df7a9e124ae05deecafecb8a9c15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([96, 4, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7c81f3934cfd18cea3338a3b99383d92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([12, 8, 288, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e08935dddf4a63eb161a81e6a638627(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da0617ae02509e1a17ba27c97d10c0cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_64168d0f0675bcbcb8ec477715d6c3ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([6, 32, 144, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_73c61909044ac443972d5c80f46298bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([960, 2, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_394a6cc4939219135b7d73fe5ba5b4ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_250fc6a2017e6735648837c560b12b56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a132bfc38db8d3333b4e0c5c6b9de51e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([2112, 2, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4f1934a193c642eb70d6ba4485ae742e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 28, 50], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c4229bafc41b0e58abcf60ce0d6aee5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f5a689168c258c15fe69b6481579b68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f5a689168c258c15fe69b6481579b68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0d76cce2fd3b6b2f535a49421eaac24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_177eb009e25462c238905c43df3c35f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b7746e46b086ba3419f6ca0e7f47ea9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d8699ce2802a60a6052f64b5ac6ec6f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4f75de05787cdabf1db95e3a7549ce6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7d50deb6297dfd58861601b0c6af47d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_52c9c79c3c080179453f9d6240944acd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99d066bbc687eb36423d43ff3122bafb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f26f36faafa162a89e56351eddfb5eff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 28, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ca2d79e0b9f33658f553c9b583883a51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f851c129732c633a4ad46847f9ea981d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2852e8ae591aaa51f17403948e649f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3fcbb27cf1a6ca832548fbfd03eea2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_22a0832aa871961e19cde036f3383213(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_127fecc3609afc53a57ca7745a5c72bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8adede6bfe920c58bd3dd09bcc23da9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([16, 64, 16, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec6775bb089152d2cc03970244df37d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 464, 16, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5258bc07ea2e66ece6950f39c698dce5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c0d76cce2fd3b6b2f535a49421eaac24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5258bc07ea2e66ece6950f39c698dce5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f1b6580916fced6c4cc127a335b45900(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 8, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db24b05619fffea2c7a84436038ac245(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b5569c5794e15bb73b384554d8b60f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f11f58ffc53f9c7dcaa83366f3ecad07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([8, 8, 288, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_15fdc23f359896960e64ca26a2131f62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 14, 25], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf097fff2285ced074acf7f447049292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a031df03b7609ac43a1d97640cb588a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_091808422d9de54c03eecb9b42bbf907(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8717537aebd882fc0e7dcab73eb251bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([240, 4, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c342418304869144f629ac2562a6072b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([4, 32, 144, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_94368a862357a41385c02e136d220ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_94368a862357a41385c02e136d220ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_94368a862357a41385c02e136d220ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4034f254aed611d77b910671adbbe3f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99d066bbc687eb36423d43ff3122bafb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f26f36faafa162a89e56351eddfb5eff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([11, 28, 28, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e44f977a3cddc510c0a37c3ab2171e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98d3299569bfaaee5972e60daa583086(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7d5b6cdb3fcd82b0304a59fc27c84e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b926274a0bd3e7f2a5d57da3daaa61fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2a1cb280a60398607de1a012a3df6c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e47eb12529731e85d7733b09781770c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_94368a862357a41385c02e136d220ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f851c129732c633a4ad46847f9ea981d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f851c129732c633a4ad46847f9ea981d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f851c129732c633a4ad46847f9ea981d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05a6c34109a99abb81db71e91f81ce13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db24b05619fffea2c7a84436038ac245(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b5569c5794e15bb73b384554d8b60f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4f47abc48c031318b6c98eb5a576d0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3fcbb27cf1a6ca832548fbfd03eea2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_22a0832aa871961e19cde036f3383213(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_696b1f769354fe0699eaa41f13a97c9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b36c038c74d61d2fa420054defee9ef9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([44, 8, 288, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0831c01ddb600fcdb69b75157742eb33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf097fff2285ced074acf7f447049292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a031df03b7609ac43a1d97640cb588a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_94368a862357a41385c02e136d220ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b30133b0479a6600d20ded5fc6322d02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a97bb28ab1b76a5b3dc57f61cfc490e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef14067aedc8d9c44ed39bab6d9b6f6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3cb6f8ac4c0ae0e48a4518ca0a7d82f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([43, 7, 7, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e2652fd30caa3cded7321b75a5ffe38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3716240e64e65b680aeccf8cd703e38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([11, 7, 7, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3d2e1b810a3a728e114c414c1ec58ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98d3299569bfaaee5972e60daa583086(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d15e6286132307e9fe2d0edff133c7c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([144, 4, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c98036a66c26af6be178d749ed197c7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08d78073bd0d59c07cafe3d49499609c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f05d358474978cf79d4f84b757d4ad5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 32, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_923bf35c87df0b04acec2ccbf96529ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b456ff49170eefcf1357174a87c7bb0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3398822304a0e847eb82023be7e870c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b409d8ea69e6cda8950e32d4fee367cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ecc3cf8a071b4ac469298293d1d5d39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e73ea87a5da733cf18e7d9023c29a895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([43, 28, 28, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d36b0e853d9c5ccab8862194696d51e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_766010bf3dcefb06d0a1619944c22f8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 232, 32, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_177eb009e25462c238905c43df3c35f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b7746e46b086ba3419f6ca0e7f47ea9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_54936e34d7b86fc2bdcd7b7e93aef2be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([43, 40, 28, 28], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_33514a27a6ef8eb725a5db76b26a4861(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b05aa68755433c110d4d272afca3eb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([4, 512, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f851c129732c633a4ad46847f9ea981d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a73a28f8cfaa760cbb624a65093be24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([4, 512, 4, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db24b05619fffea2c7a84436038ac245(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b5569c5794e15bb73b384554d8b60f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1e5882ff1ea3ad1c79599f953011c68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e104ca4cac2fe69be0e22c4cb0d4842(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f851c129732c633a4ad46847f9ea981d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_744747205f905ba40858668cdd135d13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 8, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f5a689168c258c15fe69b6481579b68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_052db243b9ffc110fbbb04d173fef33f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 116, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f35b232fd88a90c2135b720ae2c77da8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f9a49564a7562cfd0671375cf9f777a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c75fdbe4be4ea8c662710a36a3b0137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b161e70c2ae9ead6a686bc6b2d118118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bf097fff2285ced074acf7f447049292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f52ff905f84908a506c0d2c5a719f8d0
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a031df03b7609ac43a1d97640cb588a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54be9492f0763127dca1a4c6d782b47c
    def get_inputs(self):
        return [
            paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()